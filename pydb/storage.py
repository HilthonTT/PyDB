"""
Disk Manager
============
Manages a single database file as a sequence of fixed-size pages.
Handles allocation, deallocation (free-list), and raw page I/O.
 
File layout:
  [meta page 0][page 1][page 2] ...
  Meta page stores: magic(4) + version(4) + page_count(4) + free_head(4)
"""

from __future__ import annotations
 
import os
import struct
import threading
from pathlib import Path
 
from pydb import PAGE_SIZE, INVALID_PAGE, DATA_MAGIC

_META_FMT = struct.Struct("<4sIII")  # magic, version, page_count, free_head

class DiskManager:
    """Low-level page I/O against a flat database file."""
    
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
        self._fp.seek(0)
        raw = self._fp.read(_META_FMT.size)
        magic, self.version, self.page_count, self.free_head = _META_FMT.unpack(raw)
        if magic != DATA_MAGIC:
            raise ValueError("Not a PyDB data file")
        
    def _write_meta(self):
        self._fp.seek(0)
        self._fp.write(_META_FMT.pack(DATA_MAGIC, self.version,
                                       self.page_count, self.free_head))
        self._fp.flush()
        
    def read_page(self, page_id: int) -> bytes:
        with self._lock:
            self._fp.seek(page_id * PAGE_SIZE)
            data = self._fp.read(PAGE_SIZE)
            if len(data) < PAGE_SIZE:
                data += b"\x00" * (PAGE_SIZE - len(data))
            return data
        
    def write_page(self, page_id: int, data: bytes):
        assert len(data) == PAGE_SIZE
        with self._lock:
            self._fp.seek(page_id * PAGE_SIZE)
            self._fp.write(data)
            
    def flush(self):
        with self._lock:
            self._fp.flush()
            os.fsync(self._fp.fileno())
            
    def allocate_page(self) -> int:
        """Return a fresh page id. Reuses free-listed pages first."""
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
        """Return a page to the free list."""
        with self._lock:
            self._fp.seek(page_id * PAGE_SIZE)
            self._fp.write(struct.pack("<I", self.free_head))
            self.free_head = page_id
            self._write_meta()
 
    def close(self):
        self._fp.close()
