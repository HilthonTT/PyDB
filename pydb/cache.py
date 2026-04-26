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
 
from pydb import PAGE_SIZE, LRU_K, BUFFER_POOL_CAP
from pydb.page import SlottedPage
from pydb.storage import DiskManager

class _Frame:
    __slots__ = ("page_id", "page", "dirty", "pin_count", "access_ts")
    
    def __init__(self, page_id: int, page: SlottedPage):
        self.page_id   = page_id
        self.page      = page
        self.dirty     = False
        self.pin_count = 0
        self.access_ts: list[float] = [time.monotonic()]  # last K timestamps
 
class BufferPool:
    """Fixed-size buffer pool with LRU-K page replacement."""
    
    def __init__(self, disk: DiskManager, capacity: int = BUFFER_POOL_CAP, k: int = LRU_K):
        self._disk = disk
        self._cap  = capacity
        self._k    = k
        self._lock = threading.Lock()
        self._frames: dict[int, _Frame] = {}  # page_id -> frame
        
    def fetch_page(self, page_id: int) -> SlottedPage:
        """Return a *pinned* page.  Caller must call unpin() when done."""
        with self._lock:
            if page_id in self._frames:
                f = self._frames[page_id]
                f.pin_count = 1
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
        with self._lock:
            f = self._frames.get(page_id)
            if f is None:
                return
            if dirty:
                f.dirty = True
            f.pin_count = max(0, f.pin_count - 1)
            
    def mark_dirty(self, page_id: int):
        with self._lock:
            f = self._frames.get(page_id)
            if f:
                f.dirty = True
                
    def flush_page(self, page_id: int):
        with self._lock:
            f = self._frames.get(page_id)
            if f and f.dirty:
                self._disk.write_page(page_id, f.page.to_bytes())
                f.dirty = False
                
    def flush_all(self):
        with self._lock:
            for f in self._frames.values():
                if f.dirty:
                    self._disk.write_page(f.page_id, f.page.to_bytes())
                    f.dirty = False
            self._disk.flush()
            
    def new_page(self) -> SlottedPage:
        """Allocate a fresh page, pin it, and return it."""
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
        with self._lock:
            f = self._frames.pop(page_id, None)
        self._disk.deallocate_page(page_id)
        
    def _touch(self, f: _Frame):
        now = time.monotonic()
        f.access_ts.append(now)
        if len(f.access_ts) > self._k:
            f.access_ts = f.access_ts[-self._k:]
    
    def _backward_k_dist(self, f: _Frame) -> float:
        """Lower = more recently accessed at depth K → keep longer."""
        if len(f.access_ts) < self._k:
            return float("-inf")            # evict under-accessed pages first
        return f.access_ts[-self._k]        # K-th most recent access time
    
    def _evict(self):
        """Evict one frame.  Must be called while holding _lock."""
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
            self._disk.write_page(victim.page_id, victim.page.to_bytes())
        del self._frames[victim.page_id]
