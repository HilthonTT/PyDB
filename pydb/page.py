"""
Slotted-page storage format
============================
 
Page layout (PAGE_SIZE bytes):
┌──────────────────────────────────┐  offset 0
│ Page Header  (HEADER_SIZE bytes) │
│  page_id      : u32              │
│  page_type    : u8               │  0=DATA 1=BTREE_INTERNAL 2=BTREE_LEAF 3=OVERFLOW 4=FREE
│  num_slots    : u16              │
│  free_offset  : u16              │  start of free space (grows →)
│  free_end     : u16              │  end   of free space (grows ←)
│  overflow_pid : u32              │  next overflow page (INVALID if none)
│  lsn          : u64              │  log sequence number
│  _pad         : 1 byte           │
├──────────────────────────────────┤
│ Slot directory (grows →)         │
│  slot[0]: (offset:u16, len:u16)  │
│  slot[1]: ...                    │
├──────────────────────────────────┤
│         ── free space ──         │
├──────────────────────────────────┤
│ Record payloads (grow ←)         │
│  ...                             │
└──────────────────────────────────┘
"""
 
from __future__ import annotations

import struct
from enum import IntEnum
from typing import Optional
 
from pydb import PAGE_SIZE, HEADER_SIZE, SLOT_ENTRY_SIZE, INVALID_PAGE

class PageType(IntEnum):
  DATA = 0
  BTREE_INTERNAL = 1
  BTREE_LEAF = 2
  OVERFLOW = 3
  FREE = 4
  
_HDR_FMT  = struct.Struct("<IBHHHIQB")    # 4+1+2+2+2+4+8+1 = 24
_SLOT_FMT = struct.Struct("<HH")          # offset(u16), length(u16)

class RID:
  """Row identifier: (page_id, slot_index)."""
  __slots__ = ("page_id", "slot_idx")
  
  def __init__(self, page_id: int, slot_idx: int):
    self.page_id = page_id
    self.slot_idx = slot_idx
    
  def __repr__(self):
    return f"RID({self.page_id},{self.slot_idx})"
  
  def __eq__(self, value):
    return isinstance(value, RID) and self.page_id == value.page_id and self.slot_idx == value.slot_idx
  
  def __hash__(self):
    return hash((self.page_id, self.slot_idx))
  
  def to_bytes(self) -> bytes:
    return struct.pack("<IH", self.page_id, self.slot_idx)
  
  @classmethod
  def from_bytes(cls, b: bytes) -> "RID":
    pid, si = struct.unpack("<IH", b[:6])
    return cls(pid, si)
  
class SlottedPage:
    """
    Fixed-size page with a slot directory for variable-length records.
 
    Records are appended towards lower addresses; the slot directory
    grows towards higher addresses.  Deleted slots are tombstoned
    (offset=0, len=0) and reclaimed on compaction.
    """
    
    def __init__(self, page_id: int = 0, page_type: PageType = PageType.DATA, buf: Optional[bytearray] = None):
      if buf is not None:
        assert len(buf) == PAGE_SIZE
        self._buf = buf
        self._read_header()
      else:
        self._buf = bytearray(PAGE_SIZE)
        self.page_id      = page_id
        self.page_type    = page_type
        self.num_slots    = 0
        self.free_offset  = HEADER_SIZE          # first byte after header
        self.free_end     = PAGE_SIZE             # last+1 byte before records
        self.overflow_pid = INVALID_PAGE
        self.lsn          = 0
        self._write_header()
        
    def _read_header(self):
       (self.page_id, pt, self.num_slots, self.free_offset,
         self.free_end, self.overflow_pid, self.lsn, _
       ) = _HDR_FMT.unpack_from(self._buf, 0)
       self.page_type = PageType(pt)
    
    def _write_header(self):
      _HDR_FMT.pack_into(self._buf, 0,
                        self.page_id, int(self.page_type),
                        self.num_slots, self.free_offset,
                        self.free_end, self.overflow_pid, self.lsn, 0)
      