"""
Disk Manager
============
 
Overview
--------
The disk manager is the lowest layer of the storage stack — it owns
the physical database file and provides page-granularity I/O.  Every
other component accesses the file *exclusively* through this class,
so it is the single source of truth for:
 
* **Page addressing** — translating a page id (a simple integer) into
  a byte offset within the file (``page_id * PAGE_SIZE``).
* **Page allocation** — handing out fresh page ids to the buffer pool,
  B+Tree, and heap manager.
* **Page deallocation** — returning unused pages to a free list so
  they can be reused without growing the file.
 
File layout
~~~~~~~~~~~
The database file is a flat sequence of ``PAGE_SIZE``-byte (4096)
blocks::
 
    ┌──────────┬──────────┬──────────┬──────────┬─────
    │  Page 0  │  Page 1  │  Page 2  │  Page 3  │ ...
    │  (meta)  │  (data)  │  (data)  │  (free)  │
    └──────────┴──────────┴──────────┴──────────┴─────
    byte 0     byte 4096  byte 8192  byte 12288
 
**Page 0** is reserved as the **meta page**.  It stores a small header
at the beginning of the block:
 
.. list-table::
   :header-rows: 1
 
   * - Field
     - Format
     - Bytes
     - Description
   * - magic
     - ``4s``
     - 4
     - File signature ``b"PYDB"`` — validates that the file is a
       PyDB data file and not random data.
   * - version
     - ``I`` (u32)
     - 4
     - Schema version (currently ``1``).  Reserved for future
       format migrations.
   * - page_count
     - ``I`` (u32)
     - 4
     - Total number of pages in the file, including the meta page.
       The next ``allocate_page`` call returns ``page_count`` and
       then increments it.
   * - free_head
     - ``I`` (u32)
     - 4
     - Head of the free-page linked list.  ``INVALID_PAGE``
       (``0xFFFFFFFF``) if the list is empty.
 
The remaining bytes of page 0 are unused padding.
 
Free-list design
~~~~~~~~~~~~~~~~
Deallocated pages form an intrusive singly-linked list.  When a page
is freed, the first 4 bytes of its body are overwritten with the
current ``free_head``, and ``free_head`` is updated to point to the
newly freed page.  Allocation pops from the head of this list before
falling back to extending the file.  This gives O(1) alloc/dealloc
with no auxiliary data structures.
 
Concurrency
~~~~~~~~~~~
A single ``threading.Lock`` serialises all I/O.  This is acceptable
because the ``BufferPool`` above us batches most reads into cache
hits, so the disk manager only sees cold-miss reads and dirty-page
flushes — both of which are inherently I/O-bound anyway.
 
The file is opened with ``buffering=0`` (unbuffered) so that
``write`` calls go straight to the OS page cache.  An explicit
``os.fsync`` in ``flush()`` is required for true durability.
"""

from __future__ import annotations
 
import os
import struct
import threading
from pathlib import Path
 
from pydb import PAGE_SIZE, INVALID_PAGE, DATA_MAGIC

_META_FMT = struct.Struct("<4sIII")  # magic, version, page_count, free_head
"""Meta-page header format: ``magic(4s) version(u32) page_count(u32)
free_head(u32)`` — 16 bytes, little-endian."""

class DiskManager:
    """Low-level page I/O against a flat database file.
 
    On construction the manager either opens an existing file and
    reads its meta page, or creates a new file and writes a fresh
    meta page.
 
    Parameters
    ----------
    path : str or Path
        Filesystem path to the database file.  Parent directories
        must already exist.
 
    Raises
    ------
    ValueError
        If the file exists but its magic bytes are not ``b"PYDB"``.
 
    Attributes
    ----------
    version : int
        Schema version read from the meta page (currently ``1``).
    page_count : int
        Total pages in the file.  Incremented by ``allocate_page``
        when extending the file.
    free_head : int
        Page id at the head of the free list, or ``INVALID_PAGE``
        if the list is empty.
    """
    
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        exists = self._path.exists() and self._path.stat().st_size > 0
        
        # open in r+b if exists, else create w+b then reopen r+b
        if exists:
            self._fp = open(self._path, "r+b", buffering=0)
            self._read_meta()
        else:
            self._fp = open(self._path, "w+b", buffering=0)
            self.version    = 1
            self.page_count = 1        # meta page itself
            self.free_head  = INVALID_PAGE
            self._write_meta()
            
    def _read_meta(self):
        """Read and validate the 16-byte meta header from page 0.
 
        Populates ``self.version``, ``self.page_count``, and
        ``self.free_head``.
 
        Raises
        ------
        ValueError
            If the first 4 bytes are not ``DATA_MAGIC`` (``b"PYDB"``).
        """
        self._fp.seek(0)
        raw = self._fp.read(_META_FMT.size)
        magic, self.version, self.page_count, self.free_head = _META_FMT.unpack(raw)
        if magic != DATA_MAGIC:
            raise ValueError("Not a PyDB data file")
        
    def _write_meta(self):
        """Flush the current meta-page fields to the start of the file.
 
        Called after every ``allocate_page`` or ``deallocate_page``
        to keep the on-disk state consistent.  Includes an implicit
        ``flush`` to ensure the meta page reaches the OS page cache
        immediately (durability to stable storage still requires
        ``os.fsync``, which is done by the public ``flush()`` method).
        """
        self._fp.seek(0)
        self._fp.write(_META_FMT.pack(DATA_MAGIC, self.version,
                                       self.page_count, self.free_head))
        self._fp.flush()
        
    def read_page(self, page_id: int) -> bytes:
        """Read a single ``PAGE_SIZE``-byte page from the data file.
 
        If the file is shorter than expected (e.g. a page was
        allocated but the write hasn't happened yet), the missing
        bytes are zero-filled so the caller always receives a
        complete ``PAGE_SIZE``-byte block.
 
        Parameters
        ----------
        page_id : int
            The page number.  Byte offset is ``page_id * PAGE_SIZE``.
 
        Returns
        -------
        bytes
            Exactly ``PAGE_SIZE`` bytes.
        """
        with self._lock:
            self._fp.seek(page_id * PAGE_SIZE)
            data = self._fp.read(PAGE_SIZE)
            if len(data) < PAGE_SIZE:
                data += b"\x00" * (PAGE_SIZE - len(data))
            return data
        
    def write_page(self, page_id: int, data: bytes):
        """Write a ``PAGE_SIZE``-byte page to the data file.
 
        The data is written to the OS page cache but **not** fsync'd.
        Call ``flush()`` after a batch of writes if you need durability
        (the buffer pool does this during checkpoints and commits).
 
        Parameters
        ----------
        page_id : int
            The target page number.
        data : bytes
            Exactly ``PAGE_SIZE`` bytes to write.
 
        Raises
        ------
        AssertionError
            If ``len(data) != PAGE_SIZE``.
        """
        assert len(data) == PAGE_SIZE
        with self._lock:
            self._fp.seek(page_id * PAGE_SIZE)
            self._fp.write(data)
            
    def flush(self):
        """Flush all pending writes to stable storage.
 
        Calls ``fflush`` on the Python file object *and*
        ``os.fsync`` on the underlying file descriptor, ensuring
        data reaches the physical disk (or at least the drive's
        write cache, which is the best guarantee userspace can get).
        """
        with self._lock:
            self._fp.flush()
            os.fsync(self._fp.fileno())
            
    def allocate_page(self) -> int:
        """Return a fresh page id, reusing free-listed pages first.
 
        Allocation strategy:
 
        1. If ``free_head != INVALID_PAGE``, pop the head of the
           free list.  Read the first 4 bytes of the free page to
           find the *next* free page, update ``free_head``, and
           return the popped page id.
        2. Otherwise, extend the file: return the current
           ``page_count`` and increment it.  The file is extended
           by writing ``PAGE_SIZE`` zero bytes at the new offset.
 
        In both cases the meta page is flushed so that the updated
        ``page_count`` / ``free_head`` survives a crash.
 
        Returns
        -------
        int
            A page id ready for use.  The page's contents are
            undefined (may contain stale data from a previous
            occupant) — the caller should overwrite it entirely.
        """
        with self._lock:
            if self.free_head != INVALID_PAGE:
                pid = self.free_head
                # read free page to get next pointer
                self._fp.seek(pid * PAGE_SIZE)
                raw = self._fp.read(4)
                if len(raw) >= 4:
                    self.free_head = struct.unpack("<I", raw)[0]
                else:
                    self.free_head = INVALID_PAGE
                self._write_meta()
                return pid
            
            pid = self.page_count
            self.page_count += 1
            # extend file
            self._fp.seek(pid * PAGE_SIZE)
            self._fp.write(b"\x00" * PAGE_SIZE)
            self._write_meta()
            return pid
    
    def deallocate_page(self, page_id: int):
        """Return a page to the free list.
 
        Overwrites the first 4 bytes of the page with the current
        ``free_head``, then updates ``free_head`` to point to this
        page.  The rest of the page's content is left as-is (it
        will be overwritten when the page is next allocated).
 
        Parameters
        ----------
        page_id : int
            The page to free.  Must not be page 0 (the meta page).
            The caller is responsible for ensuring no other component
            still references this page (i.e. it must be unpinned and
            removed from the buffer pool first).
        """
        with self._lock:
            self._fp.seek(page_id * PAGE_SIZE)
            self._fp.write(struct.pack("<I", self.free_head))
            self.free_head = page_id
            self._write_meta()
 
    def close(self):
        """Close the underlying file handle.
 
        After ``close()`` all other methods will raise an
        ``OSError``.  The ``Database`` engine calls this during
        its own ``close()`` sequence, after the buffer pool has
        been flushed.
        """
        self._fp.close()
