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

# Struct formats
_HDR_FMT  = struct.Struct("<IBHHHIQB")    # 4+1+2+2+2+4+8+1 = 24
_SLOT_FMT = struct.Struct("<HH")          # offset(u16), length(u16)

# Record ID
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
      
    @property
    def free_space(self) -> int:
      """Gets the available free space.

      Returns:
          int: The free space representation as an integer.
      """
      return self.free_end - self.free_offset
    
    def _slot_dir_offset(self, idx: int) -> int:
      """Returns the offset for the directory.

      Args:
          idx (int): The current index.

      Returns:
          int: The offset representation as an integer.
      """
      return HEADER_SIZE + idx * SLOT_ENTRY_SIZE
    
    def _read_slot(self, idx: int) -> tuple[int, int]:
      """Reads the slot entry

      Args:
          idx (int): The index to which we access an entry.

      Returns:
          tuple[int, int]: Return a tuple containing unpacked values.
      """
      offset = self._slot_dir_offset(idx)
      return _SLOT_FMT.unpack_from(self._buf, offset)
    
    def _write_slot(self, idx: int, offset: int, length: int):
      """Writes the entry slot into the slotted page's buffer.

      Args:
          idx (int): The index of the entry.
          offset (int): The offset of the entry.
          length (int): The length of the entry.
      """
      off = self._slot_dir_offset(idx)
      _SLOT_FMT.pack_into(self._buf, off, offset, length)
      
    def insert(self, data: bytes) -> Optional[int]:
      """Insert *data* into the page. Returns slot index or None if full.

      Args:
          data (bytes): The data to be inserted into the page.

      Returns:
          Optional[int]: Returns slot index or None if full.
      """
      needed = len(data) + SLOT_ENTRY_SIZE
      if needed > self.free_space:
        return None
      
      # allocate record space from the end.
      rec_offset = self.free_end - len(data)
      self._buf[rec_offset:rec_offset + len(data)] = data
      
      # find a tombstoned slot or append
      slot_idx = None
      for i in range(self.num_slots):
        so, sl = self._read_slot(i)
        if so == 0 and sl == 0:
          slot_idx = i
          break
        
        if slot_idx is None:
          slot_idx = self.num_slots
          self.num_slots += 1
          self.free_offset = HEADER_SIZE + self.num_slots * SLOT_ENTRY_SIZE
          
        self._write_slot(i, rec_offset, len(data))
        self.free_end = rec_offset
        self._write_header()
        return slot_idx
      
    def read(self, slot_idx: int) -> Optional[bytes]:
      """Read record at *slot_idx*. Returns None if deleted / out of range.

      Args:
          slot_idx (int): The slot entry index to read.

      Returns:
          Optional[bytes]: Returns the bytes or None if deleted / out of range.
      """
      if slot_idx >= self.num_slots:
        return None
      
      offset, length = self._read_slot(slot_idx)
      if offset == 0 and length == 0:
        return None
      
      return bytes(self._buf[offset:offset + length])
    
    def delete(self, slot_idx: int) -> bool:
      """Tombstone a slot. Returns True on success.

      Args:
          slot_idx (int): The slot entry index to delete.

      Returns:
          bool: Returns True on success, otherwise False.
      """
      if slot_idx >= self.num_slots:
        return False
      
      offset, length = self._read_slot(slot_idx)
      if offset == 0 and length == 0:
        return False
      
      # Removes the entry from the buffer.
      self._write_slot(slot_idx, 0, 0)
      self._write_header()
      return True
    
    def update(self, slot_idx: int, data: bytes) -> bool:
      """Update in place if fits return True, else return False (caller should delete + reinsert).

      Args:
          slot_idx (int): The slot entry to update in place.
          data (bytes): The data which we will use to update the entry.

      Returns:
          bool: Returns True if fits, else False
      """
      if slot_idx >= self.num_slots:
        return False
      
      offset, length = self._read_slot(slot_idx)
      if offset == 0 and length == 0:
        return False
      
      if len(data) > length:
        return False
      
      self._buf[offset:offset + len(data)] = data
      if len(data) < length:
          # zero remainder
          self._buf[offset + len(data):offset + length] = b"\x00" * (length - len(data))
      self._write_slot(slot_idx, offset, len(data))
      self._write_header()
      return True
    
    def compact(self):
      """Reclaim dead space by rewriting live records contiguously."""
      records: list[tuple[int, bytes]] = []
      for i in range(self.num_slots):
        offset, length = self._read_slot(i)
        if offset != 0 and length != 0:
          records.append((i, bytes(self._buf[offset:offset + length])))
          
      # reset data area
      self.free_end = PAGE_SIZE
      for slot_idx, data in records:
        self.free_end -= len(data)
        self._buf[self.free_end:self.free_end + len(data)] = data
        self._write_slot(slot_idx, self.free_end, len(data))
      self._write_header()
      
    def to_bytes(self) -> bytes:
      """Converts the slotted page into bytes.
      
      Returns:
          bytes: The bytes representation of the slotted page.
      """
      return bytes(self._buf)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> SlottedPage:
      """Converts the bytes into a slotted page.

      Args:
          data (bytes): The buffer of the slotted page.

      Returns:
          SlottedPage: The slotted page instance.
      """
      return cls(buf=bytearray(data))
    
    def iter_records(self):
      """Yield (slot_idx, record_bytes) for every live record."""
      for i in range(self.num_slots):
          off, length = self._read_slot(i)
          if off != 0 or length != 0:
              yield i, bytes(self._buf[off:off + length])
