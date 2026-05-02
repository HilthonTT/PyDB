"""
K-way External Merge Sort
=========================
 
Overview
--------
When an ``ORDER BY`` clause cannot be satisfied by an index scan, the
executor must sort the result set.  If the data fits in memory, a
simple in-memory sort is used.  But for arbitrarily large results,
we need an **external sort** — one that spills to disk when memory
is exhausted.
 
This module implements the textbook **K-way external merge sort**
algorithm in two phases:
 
Phase 1 — Run generation
~~~~~~~~~~~~~~~~~~~~~~~~~
The input iterator is consumed in chunks of ``max_rows_in_memory``
rows.  Each chunk is sorted in-memory using Python's built-in
Timsort (``list.sort``) and written to a temporary file on disk.
Each such file is called a **sorted run**.
 
Phase 2 — Multi-pass K-way merge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If there are more runs than the fan-in *K*, we merge *K* runs at a
time into a new, larger run, repeating until only one run remains.
Each merge pass uses a **min-heap** (``heapq``) to efficiently
select the smallest element across all *K* input streams.
 
With *N* total rows, *M* rows per run, and fan-in *K*:
 
* Phase 1 produces ``ceil(N / M)`` runs.
* Phase 2 requires ``ceil(log_K(N / M))`` merge passes.
* Each pass reads and writes every row once → total I/O is
  ``O(N · log_K(N / M))``.
 
Descending sort is handled by wrapping keys in a ``_NegKey``
comparator that reverses all comparisons, so the min-heap
effectively becomes a max-heap.
 
Temporary files
~~~~~~~~~~~~~~~
Sorted runs are serialised with Python's ``pickle`` protocol and
stored in the system's temp directory (or a custom ``tmp_dir``).
Files are cleaned up after each merge pass and on ``cleanup()``.
"""
 
from __future__ import annotations
 
import heapq
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator
 
def _default_key(row):
    """Identity key function — sort by the row itself."""
    return row

class ExternalMergeSorter:
    """K-way external merge sort for arbitrarily large datasets.
 
    Parameters
    ----------
    key_func : callable
        Extracts the sort key from each row, like the *key* argument
        to Python's ``sorted()``.  Defaults to identity.
    reverse : bool
        If ``True``, sort in descending order.
    max_rows_in_memory : int
        Maximum rows per sorted run before spilling to disk.
        Larger values produce fewer runs (fewer merge passes) at the
        cost of more memory.
    k : int
        Fan-in for each merge pass — how many runs are merged
        simultaneously.  Higher values reduce the number of merge
        passes but increase heap overhead.  Minimum 2.
    tmp_dir : str, Path, or None
        Directory for temporary run files.  Defaults to the system
        temp directory (``/tmp`` on Linux).
 
    Examples
    --------
    Sort 1 million integers with at most 100K in memory::
 
        sorter = ExternalMergeSorter(max_rows_in_memory=100_000, k=8)
        result = list(sorter.sort(iter(range(1_000_000, 0, -1))))
        assert result == list(range(1, 1_000_001))
    """
    
    def __init__(
        self,
        key_func: Callable = _default_key,
        reverse: bool = False,
        max_rows_in_memory: int = 10_000,
        k: int = 16,
        tmp_dir: str | Path | None = None):
        self._key = key_func
        self._reverse = reverse
        self._mem = max_rows_in_memory
        self._k = max(2, k)
        self._tmp = Path(tmp_dir) if tmp_dir else None
        self._run_files: list[Path] = []
    
    def sort(self, rows: Iterator) -> Iterator:
        """Sort *rows* and return a sorted iterator.
 
        This is the main entry point.  The input iterator is consumed
        lazily during Phase 1 (run generation).  The output iterator
        streams results from the final merged run, so memory usage
        stays bounded.
 
        Parameters
        ----------
        rows : Iterator
            The unsorted input.  Consumed once, front to back.
 
        Returns
        -------
        Iterator
            The sorted output.  Yields rows one at a time.
        """
        self._run_files.clear()
        
        # Phase 1 - create sorted runs
        buf: list = []
        for row in rows:
            buf.append(row)
            if len(buf) >= self._mem:
                self._flush_run(buf)
                buf = []
        
        if buf:
            self._flush_run(buf)
            
        if not self._run_files:
            return iter([])
        
        # Everything fit in one run — stream it back directly.
        if len(self._run_files) == 1:
            return self._read_run(self._run_files[0])
        
        # Phase 2 - multi-pass K-way merge
        runs = list(self._run_files)
        try:
            while len(runs) > 1:
                next_runs: list[Path] = []
                for i in range(0, len(runs), self._k):
                    group = runs[i:i + self._k]
                    if len(group) == 1:
                        next_runs.append(group[0])
                    else:
                        merged = self._merge_runs(group)
                        next_runs.append(merged)
                        for p in group:
                            p.unlink(missing_ok=True)
                runs = next_runs
        except Exception:
            for p in runs:
                p.unlink(missing_ok=True)
            raise

        return self._read_run(runs[0])
    
    def cleanup(self):
        """Delete all temporary run files.
 
        Called automatically during merge passes for consumed runs.
        Call explicitly if the sort is abandoned before completion.
        """
        for p in self._run_files:
            p.unlink(missing_ok=True)
            
    def _flush_run(self, buf: list):
        """Sort *buf* in memory and write it to a new temp file."""
        buf.sort(key=self._key, reverse=self._reverse)
        path = self._make_tmp()
        with open(path, "wb") as f:
            for row in buf:
                pickle.dump(row, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._run_files.append(path)
        
    def _make_tmp(self) -> Path:
        """Create a new temp file and return its path."""
        fd, name = tempfile.mkstemp(suffix=".run", dir=self._tmp)
        os.close(fd)
        return Path(name)

    def _read_run(self, path: Path) -> Iterator:
        """Yield rows from a pickled run file, front to back."""
        with open(path, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
        
    def _merge_runs(self, paths: list[Path]) -> Path:
        """Merge multiple sorted run files into one using a min-heap.
 
        Each heap entry is ``(sort_key, source_index, row)`` so that
        ties are broken by source index (stable sort within a pass).
        """
        iters = [self._read_run(p) for p in paths]
        out_path = self._make_tmp()
        
        heap: list[tuple] = []
        for idx, it in enumerate(iters):
            row = next(it, None)
            if row is not None:
                k = self._key(row)
                if self._reverse:
                    entry = (_NegKey(k), idx, row)
                else:
                    entry = (k, idx, row)
                heapq.heappush(heap, entry)
                
        with open(out_path, "wb") as f:
            while heap:
                if self._reverse:
                    _, idx, row = heapq.heappop(heap)
                else:
                    _, idx, row = heapq.heappop(heap)
                pickle.dump(row, f, protocol=pickle.HIGHEST_PROTOCOL)
                nxt = next(iters[idx], None)
                if nxt is not None:
                    k = self._key(nxt)
                    if self._reverse:
                        heapq.heappush(heap, (_NegKey(k), idx, nxt))
                    else:
                        heapq.heappush(heap, (k, idx, nxt))
 
        return out_path

class _NegKey:
    """Wrapper that reverses comparison order for descending sort.
 
    Used to turn a min-heap into a max-heap: wrapping keys in
    ``_NegKey`` makes ``heapq`` pop the *largest* key first.
    """
 
    __slots__ = ("key",)
    
    def __init__(self, key):
        self.key = key
        
    def __lt__(self, other):
        return self.key > other.key
    
    def __le__(self, other):
        return self.key >= other.key
    
    def __gt__(self, other):
        return self.key < other.key
    
    def __ge__(self, other):
        return self.key <= other.key
    
    def __eq__(self, other):
        return self.key == other.key
    