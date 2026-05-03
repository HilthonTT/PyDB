"""
Buffer Pool with LRU-K Replacement
===================================
Caches pages in memory, using an LRU-K eviction policy (K=2 by default).
 
LRU-K tracks the *K-th* most recent access timestamp for each page.
On eviction we pick the frame whose K-th access is the oldest (maximally
backward K-distance).  Pages accessed fewer than K times are evicted first.
 
Features:
 - Pin counting (pinned pages are never evicted)
 - Dirty-page tracking with flush-on-evict
 - Thread-safe via a single lock (sufficient for moderate concurrency)
"""

from __future__ import annotations
 
import threading
import time
from typing import Optional
 
from pydb import LRU_K, BUFFER_POOL_CAP
from pydb.page import SlottedPage
from pydb.storage import DiskManager

class _Frame:
    """A single slot (frame) inside the buffer pool.
 
    Each frame holds exactly one page and the bookkeeping metadata
    the pool needs to manage it:
 
    Attributes
    ----------
    page_id : int
        The on-disk page number currently loaded into this frame.
    page : SlottedPage
        The in-memory page object.  All users that ``fetch_page``
        the same ``page_id`` share this exact object.
    dirty : bool
        ``True`` if the page has been modified in memory but not yet
        written back to disk.  The pool flushes dirty frames on eviction
        and on explicit ``flush_page`` / ``flush_all`` calls.
    pin_count : int
        The number of outstanding users of this frame.  A page with
        ``pin_count > 0`` is **never** considered for eviction.
        Incremented by ``fetch_page``, decremented by ``unpin``.
    access_ts : list[float]
        Rolling window of the last *K* access timestamps (from
        ``time.monotonic``).  Used by the LRU-K replacement algorithm
        to compute the backward K-distance.
 
        Example with K = 2 and three accesses at t=1.0, t=3.0, t=7.0::
 
            access_ts = [3.0, 7.0]   # only the last K are kept
 
        The backward K-distance is ``access_ts[0]`` = 3.0 (the 2nd
        most recent access).  A *smaller* value means the page hasn't
        been touched deeply in a while → more likely to be evicted.
    """
    
    __slots__ = ("page_id", "page", "dirty", "pin_count", "access_ts")
    
    def __init__(self, page_id: int, page: SlottedPage):
        self.page_id   = page_id
        self.page      = page
        self.dirty     = False
        self.pin_count = 0
        self.access_ts: list[float] = [time.monotonic()]  # last K timestamps
 
class BufferPool:
    """Fixed-size buffer pool with LRU-K page replacement.
 
    The pool manages a dictionary of ``_Frame`` objects, keyed by
    page id.  When the number of frames reaches ``capacity`` and a
    new page is requested, the pool evicts the frame with the
    smallest backward K-distance (see ``_backward_k_dist``).
 
    Parameters
    ----------
    disk : DiskManager
        The underlying disk I/O layer.  The pool calls
        ``disk.read_page`` on cache misses and ``disk.write_page``
        when flushing dirty pages.
    capacity : int
        Maximum number of page frames to keep in memory.
        Defaults to ``BUFFER_POOL_CAP`` (1024 frames = 4 MiB
        with 4 KiB pages).
    k : int
        The *K* parameter for LRU-K.  Defaults to ``LRU_K`` (2).
        Higher values give the algorithm more history to work with,
        but K = 2 is the standard choice in the literature.
 
    Raises
    ------
    RuntimeError
        If ``_evict`` is called but every frame in the pool is
        pinned.  This indicates a pin leak in the caller.
 
    Examples
    --------
    Basic fetch / modify / unpin cycle::
 
        pool = BufferPool(disk, capacity=256)
 
        # Fetch pins the page — it won't be evicted while we hold it.
        page = pool.fetch_page(page_id=5)
        slot = page.insert(b"hello")
 
        # Unpin and mark dirty so it gets flushed to disk.
        pool.unpin(5, dirty=True)
 
    Allocating a brand-new page::
 
        page = pool.new_page()        # pinned + dirty
        page.insert(some_record)
        pool.unpin(page.page_id, dirty=True)
    """
    
    def __init__(self, disk: DiskManager, capacity: int = BUFFER_POOL_CAP, k: int = LRU_K,
                 wal=None):
        self._disk = disk
        self._cap  = capacity
        self._k    = k
        self._wal  = wal
        self._lock = threading.Lock()
        self._frames: dict[int, _Frame] = {}  # page_id -> frame
        
    def fetch_page(self, page_id: int) -> SlottedPage:
        """Fetch a page by id, loading it from disk on a cache miss.
 
        If the page is already cached, its pin count is incremented
        and its access history is updated (``_touch``).  If it is not
        cached, the pool may evict another frame to make room, then
        reads the raw bytes from the ``DiskManager`` and deserialises
        them into a ``SlottedPage``.
 
        The returned page is **pinned** — the caller **must** call
        ``unpin(page_id)`` when it is finished with the page.
        Failing to unpin will eventually exhaust the pool.
 
        Parameters
        ----------
        page_id : int
            The on-disk page number to fetch.
 
        Returns
        -------
        SlottedPage
            The page object.  This is a shared mutable reference — all
            concurrent users of the same page_id see the same object.
        """
        with self._lock:
            if page_id in self._frames:
                f = self._frames[page_id]
                f.pin_count += 1
                self._touch(f)
                return f.page
            
            # not cached - need to load from disk
            if len(self._frames) >= self._cap:
                self._evict()
                
            raw = self._disk.read_page(page_id)
            page = SlottedPage.from_bytes(raw)
            page.page_id = page_id
 
            f = _Frame(page_id, page)
            f.pin_count = 1
            self._frames[page_id] = f
            return page
        
    def unpin(self, page_id: int, dirty: bool = False):
        """Release a pin on a page, optionally marking it dirty.
 
        Decrements the frame's ``pin_count`` (clamped to 0).  Once the
        count reaches 0 the frame becomes eligible for eviction.
 
        Parameters
        ----------
        page_id : int
            The page to unpin.  If the page is not in the pool (e.g.
            already evicted), this is a silent no-op.
        dirty : bool
            If ``True``, the frame is flagged dirty so that it will be
            written back to disk before eviction or on an explicit
            flush.  This is a sticky flag — once set, it stays ``True``
            until the page is flushed.
        """
        with self._lock:
            f = self._frames.get(page_id)
            if f is None:
                return
            if dirty:
                f.dirty = True
            f.pin_count = max(0, f.pin_count - 1)
            
    def mark_dirty(self, page_id: int):
        """Mark a cached page as dirty without changing its pin count.
 
        Useful when a higher-level component (e.g. the WAL) knows a
        page has been modified but doesn't hold the pin itself.
 
        Parameters
        ----------
        page_id : int
            The page to mark.  Silent no-op if not cached.
        """
        with self._lock:
            f = self._frames.get(page_id)
            if f:
                f.dirty = True
                
    def flush_page(self, page_id: int):
        """Write a single dirty page back to disk.
 
        If the page is not dirty (or not in the pool), this is a
        no-op.  After flushing, the dirty flag is cleared but the
        page remains in the pool — flushing does **not** evict.
 
        Parameters
        ----------
        page_id : int
            The page to flush.
        """
        with self._lock:
            f = self._frames.get(page_id)
            if f and f.dirty:
                self._disk.write_page(page_id, f.page.to_bytes())
                f.dirty = False
                
    def flush_all(self):
        """Write every dirty page to disk and call ``fsync``.
 
        This is the "checkpoint" operation: after ``flush_all``
        returns, every modification that was made in-memory is
        guaranteed to be durable on the underlying storage device.
        The pool itself is **not** cleared — all frames remain
        cached and usable.
        """
        with self._lock:
            for f in self._frames.values():
                if f.dirty:
                    self._disk.write_page(f.page_id, f.page.to_bytes())
                    f.dirty = False
            self._disk.flush()
            
    def new_page(self) -> SlottedPage:
        """Allocate a fresh page from the disk manager and cache it.
 
        The page is returned **pinned** (``pin_count = 1``) and
        **dirty** (it has never been written yet).  The caller must
        unpin it when done.
 
        Returns
        -------
        SlottedPage
            A new, empty page with a freshly assigned ``page_id``.
            The id comes from the ``DiskManager``'s free list (if a
            previously deallocated page is available) or by extending
            the data file.
        """
        pid = self._disk.allocate_page()
        with self._lock:
            if len(self._frames) >= self._cap:
                self._evict()
            page = SlottedPage(page_id=pid)
            f = _Frame(pid, page)
            f.pin_count = 1
            f.dirty = True
            self._frames[pid] = f
            return page
    
    def delete_page(self, page_id: int):
        """Remove a page from the pool and return it to the disk free list.
 
        The frame (if cached) is discarded without flushing — the
        caller is asserting that the page's contents are no longer
        needed.  The ``DiskManager`` adds the page to its free list
        so it can be re-used by a future ``allocate_page`` call.
 
        Parameters
        ----------
        page_id : int
            The page to delete.  If not currently cached, only the
            disk-level deallocation happens.
        """
        with self._lock:
            self._frames.pop(page_id, None)
        self._disk.deallocate_page(page_id)
        
    def _touch(self, f: _Frame):
        """Record a new access timestamp for the LRU-K algorithm.
 
        Appends the current ``time.monotonic()`` value to the frame's
        ``access_ts`` list and trims it to the last *K* entries.
 
        This is called on every cache **hit** inside ``fetch_page``.
        Cache misses don't call ``_touch`` because the ``_Frame``
        constructor already seeds ``access_ts`` with one timestamp.
 
        Parameters
        ----------
        f : _Frame
            The frame that was just accessed.
        """
        now = time.monotonic()
        f.access_ts.append(now)
        if len(f.access_ts) > self._k:
            f.access_ts = f.access_ts[-self._k:]
    
    def _backward_k_dist(self, f: _Frame) -> float:
        """Compute the backward K-distance for a frame.
 
        The backward K-distance is the timestamp of the K-th most
        recent access.  A **smaller** value means the page's deep
        history is older → it is a better eviction candidate.
 
        Special case: if the page has been accessed fewer than K
        times, it has *infinite* backward K-distance in the original
        paper.  We return ``-inf`` instead so that these "cold" pages
        sort below *all* pages with full history, making them the
        first to be evicted.  This is the scan-resistance property:
        a sequential scan that touches each page only once will fill
        the pool with cold pages that are evicted before any page
        with established access depth.
 
        Parameters
        ----------
        f : _Frame
            The frame to evaluate.
 
        Returns
        -------
        float
            The K-th most recent ``time.monotonic()`` value, or
            ``-inf`` if the frame has fewer than K accesses.
 
        Examples
        --------
        With K = 2::
 
            access_ts = [3.0, 7.0]  →  backward_k_dist = 3.0
            access_ts = [7.0]       →  backward_k_dist = -inf  (cold)
        """
        if len(f.access_ts) < self._k:
            return float("-inf")            # evict under-accessed pages first
        return f.access_ts[-self._k]        # K-th most recent access time
    
    def _evict(self):
        """Choose and remove one frame to make room for a new page.
 
        Selection algorithm:
 
        1. Iterate all frames, skip any with ``pin_count > 0``.
        2. Pick the frame with the **smallest** backward K-distance
           (i.e. the one whose K-th access is the most distant in
           the past — or ``-inf`` if it hasn't been accessed K times).
        3. If the victim is dirty, flush it to disk first.
        4. Remove the victim from ``_frames``.
 
        Must be called while holding ``_lock``.
 
        Raises
        ------
        RuntimeError
            If every frame in the pool is pinned and no victim can
            be selected.  This always indicates a bug in the caller
            (missing ``unpin`` calls).
        """
        victim: Optional[_Frame] = None
        victim_dist = float("inf")
 
        for f in self._frames.values():
            if f.pin_count > 0:
                continue
            dist = self._backward_k_dist(f)
            if dist < victim_dist:
                victim_dist = dist
                victim = f
 
        if victim is None:
            raise RuntimeError("Buffer pool full: all pages are pinned")
 
        if victim.dirty:
            if self._wal:
                self._wal.flush()
            self._disk.write_page(victim.page_id, victim.page.to_bytes())
        del self._frames[victim.page_id]
