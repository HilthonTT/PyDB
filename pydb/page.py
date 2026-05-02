"""
Slotted-Page Storage Format
============================
 
Overview
--------
The slotted page is the fundamental on-disk data structure.  Every piece
of persistent data — heap records, B+Tree nodes, overflow chains — lives
inside a ``SlottedPage``.  Each page is a fixed ``PAGE_SIZE`` (4096 bytes)
block that can hold a variable number of variable-length records.
 
Page layout
~~~~~~~~~~~
The design follows the classic slotted-page scheme from database
textbooks (Ramakrishnan & Gehrke, *Database Management Systems*).
Three regions share the fixed-size buffer::
 
    ┌──────────────────────────────────┐  offset 0
    │ Page Header  (HEADER_SIZE = 24)  │
    ├──────────────────────────────────┤  offset 24
    │ Slot directory  (grows →)        │
    │  slot[0]: (offset:u16, len:u16)  │
    │  slot[1]: ...                    │
    ├──────────────────────────────────┤  ← free_offset
    │          ── free space ──        │
    ├──────────────────────────────────┤  ← free_end
    │ Record payloads  (grow ←)        │
    │  ...                             │
    └──────────────────────────────────┘  offset PAGE_SIZE
 
* The **slot directory** starts right after the header and grows
  *forward* (toward higher addresses).  Each 4-byte entry holds the
  byte offset and byte length of one record.
* **Record payloads** are written from the *end* of the page and grow
  *backward* (toward lower addresses).
* The **free space** is the gap between the end of the slot directory
  (``free_offset``) and the start of the record region (``free_end``).
 
This opposing-growth scheme means we never need to move records when
new slots are added, and we never need to move slot entries when new
records are written.
 
Header format
~~~~~~~~~~~~~
Packed as ``struct.Struct("<IBHHHIQB")`` — 24 bytes total::
 
    page_id       u32   The page's position in the data file.
    page_type     u8    Discriminant for DATA / BTREE_INTERNAL /
                        BTREE_LEAF / OVERFLOW / FREE.
    num_slots     u16   Number of slot entries (including tombstones).
    free_offset   u16   First byte of free space (just past the last slot).
    free_end      u16   Last+1 byte of free space (just before the first record).
    overflow_pid  u32   Next page in an overflow chain (INVALID_PAGE if none).
    lsn           u64   Log Sequence Number — set by the WAL on each write,
                        used during crash recovery to decide whether a
                        redo/undo is needed.
    _pad          u8    Alignment padding to reach exactly 24 bytes.
 
Deletion and compaction
~~~~~~~~~~~~~~~~~~~~~~~
Deleting a record does *not* physically remove its bytes.  Instead the
slot entry is **tombstoned** — both offset and length are set to zero.
The dead space is reclaimed lazily:
 
* On the next ``insert``, if a tombstoned slot is found, it is reused
  for the new record's directory entry (the record payload itself is
  still allocated from ``free_end``).
* Calling ``compact()`` rewrites all live records contiguously from the
  end of the page, closing all gaps left by deletions.
 
Overflow pages
~~~~~~~~~~~~~~
When a record is too large to fit in a single data page, it is stored
across a chain of overflow pages.  An overflow page uses a small
8-byte sub-header (``OVERFLOW_HEADER``) immediately after the standard
page header::
 
    next_page_id   u32   Next overflow page (INVALID_PAGE if last).
    payload_len    u16   Number of payload bytes in *this* page.
    _pad           2x    Alignment.
 
The ``make_overflow_page`` / ``read_overflow_payload`` helpers handle
this encoding.
"""
 
from __future__ import annotations

import struct
from enum import IntEnum
from typing import Optional
 
from pydb import PAGE_SIZE, HEADER_SIZE, SLOT_ENTRY_SIZE, INVALID_PAGE

class PageType(IntEnum):
  """Discriminant stored in byte 4 of every page header.
 
    The executor, B+Tree, and heap scanner all check this value to
    decide how to interpret the page's contents.
 
    Members
    -------
    DATA : 0
        A heap page holding table records.
    BTREE_INTERNAL : 1
        An internal (non-leaf) B+Tree node storing keys and child
        page pointers.
    BTREE_LEAF : 2
        A leaf B+Tree node storing keys and ``RID`` values, plus
        a ``next_leaf`` pointer for range-scan chaining.
    OVERFLOW : 3
        Part of an overflow chain for records exceeding a single page.
    FREE : 4
        An unused page returned to the ``DiskManager``'s free list.
    """
  DATA = 0
  BTREE_INTERNAL = 1
  BTREE_LEAF = 2
  OVERFLOW = 3
  FREE = 4

# Struct formats
_HDR_FMT  = struct.Struct("<IBHHHIQB")    # 4+1+2+2+2+4+8+1 = 24
"""Page header: ``page_id(u32) page_type(u8) num_slots(u16)
free_offset(u16) free_end(u16) overflow_pid(u32) lsn(u64) pad(u8)``
— 24 bytes, little-endian."""

_SLOT_FMT = struct.Struct("<HH")          # offset(u16), length(u16)
"""Single slot-directory entry: ``offset(u16) length(u16)`` — 4 bytes.
An entry with ``offset == 0 and length == 0`` is a tombstone (deleted)."""

# Record ID
class RID:
  """Row Identifier — the physical address of a record on disk.
 
    A RID is the (page_id, slot_index) pair that uniquely locates a
    record inside the database file.  B+Tree leaf nodes store RIDs
    as their values, and the executor uses them to fetch, update, or
    delete individual records.
 
    The wire format is 6 bytes: ``page_id(u32) slot_idx(u16)``,
    packed little-endian.
 
    Parameters
    ----------
    page_id : int
        The page number within the data file.
    slot_idx : int
        The zero-based index into the page's slot directory.
 
    Examples
    --------
    ::
 
        rid = RID(42, 3)            # page 42, slot 3
        raw = rid.to_bytes()        # b'\\x2a\\x00\\x00\\x00\\x03\\x00'
        assert RID.from_bytes(raw) == rid
  """
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
    """Serialise to 6 bytes: ``<IH`` (page_id u32, slot_idx u16)."""
    return struct.pack("<IH", self.page_id, self.slot_idx)
  
  @classmethod
  def from_bytes(cls, b: bytes) -> "RID":
    """Deserialise from the first 6 bytes of *b*."""
    pid, si = struct.unpack("<IH", b[:6])
    return cls(pid, si)
  
class SlottedPage:
    """Fixed-size page with a slot directory for variable-length records.
 
    The page manages a flat ``PAGE_SIZE``-byte buffer (``self._buf``)
    that is read from and written to disk as an atomic unit.  All
    field access (header, slot entries, record payloads) goes through
    ``struct`` pack/unpack operations against this buffer.
 
    Construction
    ~~~~~~~~~~~~
    There are two construction paths:
 
    * **Fresh page** — ``SlottedPage(page_id=N)`` creates an empty
      page with the header initialised and all record space free.
    * **From disk** — ``SlottedPage(buf=bytearray(...))`` (or the
      classmethod ``from_bytes``) wraps an existing buffer and
      parses the header out of it.
 
    Parameters
    ----------
    page_id : int
        Page number.  Only used for fresh pages; when loading from
        a buffer the stored ``page_id`` in the header takes precedence.
    page_type : PageType
        The initial page type.  Defaults to ``DATA``.
    buf : bytearray or None
        If provided, the page wraps this buffer instead of allocating
        a new one.  Must be exactly ``PAGE_SIZE`` bytes.
 
    Attributes
    ----------
    page_id : int
        On-disk page number (from header).
    page_type : PageType
        How this page is used (from header).
    num_slots : int
        Total slot-directory entries, **including** tombstones.
    free_offset : int
        Byte offset of the first byte of free space (immediately
        after the last slot-directory entry).
    free_end : int
        Byte offset one past the end of free space (immediately
        before the first record payload).
    overflow_pid : int
        Next page in an overflow or heap chain.  ``INVALID_PAGE``
        (``0xFFFFFFFF``) if this is the last page.
    lsn : int
        Log Sequence Number, written by the WAL subsystem.
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
      """Unpack the 24-byte header from the front of ``_buf``
      into the instance attributes (``page_id``, ``page_type``, etc.).
 
      Called once during construction when wrapping an existing buffer.
      """
      (self.page_id, pt, self.num_slots, self.free_offset,
         self.free_end, self.overflow_pid, self.lsn, _
      ) = _HDR_FMT.unpack_from(self._buf, 0)
      self.page_type = PageType(pt)
    
    def _write_header(self):
      """Pack the current instance attributes back into the first
      24 bytes of ``_buf``.
 
      This must be called after **every** mutation that changes a
      header field (``num_slots``, ``free_offset``, ``free_end``,
      ``overflow_pid``, or ``lsn``), because ``to_bytes()`` simply
      returns the raw buffer — there is no lazy serialisation step.
      """
      _HDR_FMT.pack_into(self._buf, 0,
                        self.page_id, int(self.page_type),
                        self.num_slots, self.free_offset,
                        self.free_end, self.overflow_pid, self.lsn, 0)
      
    @property
    def free_space(self) -> int:
      """The number of usable bytes between the slot directory and
      the record region.

      A new record of *N* bytes needs ``N + SLOT_ENTRY_SIZE`` (4)
      free bytes: *N* for the payload written at the bottom, plus
      4 bytes for a new slot-directory entry at the top (unless a
      tombstoned slot is available for reuse, in which case only
      *N* bytes of free space are consumed).
      """
      return self.free_end - self.free_offset
    
    def _slot_dir_offset(self, idx: int) -> int:
      """Byte offset within ``_buf`` where slot *idx*'s entry begins.

      Slot entries are stored contiguously starting at byte
      ``HEADER_SIZE`` (24), each occupying ``SLOT_ENTRY_SIZE`` (4)
      bytes::

          slot 0 → bytes [24..28)
          slot 1 → bytes [28..32)
          slot N → bytes [24 + N*4 .. 24 + (N+1)*4)

      Parameters
      ----------
      idx : int
          Zero-based slot index.
      """
      return HEADER_SIZE + idx * SLOT_ENTRY_SIZE
    
    def _read_slot(self, idx: int) -> tuple[int, int]:
      """Read slot *idx* and return ``(byte_offset, byte_length)``.
 
      Both values are unsigned 16-bit integers.  A tombstoned
      (deleted) slot returns ``(0, 0)``.

      Parameters
      ----------
      idx : int
          Zero-based slot index.  Must be ``< num_slots``.
      """
      offset = self._slot_dir_offset(idx)
      return _SLOT_FMT.unpack_from(self._buf, offset)
    
    def _write_slot(self, idx: int, offset: int, length: int):
      """Write ``(offset, length)`` into slot *idx*'s directory entry.
 
      Parameters
      ----------
      idx : int
          Zero-based slot index.
      offset : int
          Byte offset of the record payload within the page buffer.
          Set to ``0`` together with ``length=0`` to tombstone.
      length : int
          Length of the record payload in bytes.
      """
      off = self._slot_dir_offset(idx)
      _SLOT_FMT.pack_into(self._buf, off, offset, length)
      
    def insert(self, data: bytes) -> Optional[int]:
      """Insert a record into the page.

      The record payload is written at the bottom of free space
      (``free_end`` moves down by ``len(data)``).  A slot-directory
      entry is either reused from a tombstoned slot or appended to
      the directory (``free_offset`` moves up by ``SLOT_ENTRY_SIZE``).

      Parameters
      ----------
      data : bytes
          The raw record bytes to store.  Maximum length is
          ``PAGE_SIZE - HEADER_SIZE - SLOT_ENTRY_SIZE`` (4068 bytes)
          for a completely empty page.

      Returns
      -------
      int or None
          The slot index where the record was placed, or ``None``
          if the page does not have enough free space.
      """
      # find a tombstoned slot to reuse
      slot_idx = None
      for i in range(self.num_slots):
        so, sl = self._read_slot(i)
        if so == 0 and sl == 0:
          slot_idx = i
          break

      # calculate space needed — no extra SLOT_ENTRY_SIZE if reusing a tombstone
      needed = len(data) + (0 if slot_idx is not None else SLOT_ENTRY_SIZE)
      if needed > self.free_space:
        return None

      # allocate record space from the end
      rec_offset = self.free_end - len(data)
      self._buf[rec_offset:rec_offset + len(data)] = data

      if slot_idx is None:
        slot_idx = self.num_slots
        self.num_slots += 1
        self.free_offset = HEADER_SIZE + self.num_slots * SLOT_ENTRY_SIZE

      self._write_slot(slot_idx, rec_offset, len(data))
      self.free_end = rec_offset
      self._write_header()
      return slot_idx
      
    def read(self, slot_idx: int) -> Optional[bytes]:
      """Read the record at slot *slot_idx*.

      Parameters
      ----------
      slot_idx : int
          Zero-based slot index.

      Returns
      -------
      bytes or None
          The record payload, or ``None`` if the slot is out of
          range or has been tombstoned (deleted).
      """
      if slot_idx >= self.num_slots:
        return None
      
      offset, length = self._read_slot(slot_idx)
      if offset == 0 and length == 0:
        return None
      
      return bytes(self._buf[offset:offset + length])
    
    def delete(self, slot_idx: int) -> bool:
      """Tombstone a record by zeroing its slot-directory entry.
 
      The actual payload bytes are **not** erased — they become
      dead space that is reclaimed when ``compact()`` runs, or
      simply ignored on all subsequent reads.

      Parameters
      ----------
      slot_idx : int
          The slot to delete.

      Returns
      -------
      bool
          ``True`` if the slot was live and is now tombstoned.
          ``False`` if it was already deleted or out of range.
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
      """Update a record in place if the new data fits.
 
      If ``len(data) <= original_length``, the payload is
      overwritten at the same offset and any leftover bytes are
      zero-filled.  The slot entry's length field is updated to
      the new (possibly shorter) size.

      If the new data is **larger** than the original, the update
      fails and returns ``False``.  The caller should then
      ``delete`` + ``insert`` to relocate the record (potentially
      on a different page).

      Parameters
      ----------
      slot_idx : int
          The slot to update.
      data : bytes
          The new record payload.

      Returns
      -------
      bool
          ``True`` if the update was performed in place.
          ``False`` if the record doesn't exist, is tombstoned,
          or the new data is too large.
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
      """
      Reclaim dead space by rewriting all live records contiguously.
      Walks the slot directory, collects every non-tombstoned record,
      then repacks them tightly from the end of the page.  Slot
      entries are updated with the new offsets; ``free_end`` is
      adjusted accordingly.
 
      This is an O(N) operation over the number of live records
      and should be called when the page has significant
      fragmentation (many deletes followed by failed inserts).
      """
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
      """
      Serialise the page to an immutable ``PAGE_SIZE``-byte blob.
 
      Flushes the header into the buffer first, then returns a
      snapshot.  This is what the ``BufferPool`` writes to disk.
      """
      return bytes(self._buf)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> SlottedPage:
      """Construct a ``SlottedPage`` from a raw ``PAGE_SIZE``-byte
      blob read from disk.

      Parameters
      ----------
      data : bytes
          Exactly ``PAGE_SIZE`` bytes.  Typically comes from
          ``DiskManager.read_page``.
      """
      return cls(buf=bytearray(data))
    
    def iter_records(self):
      """Yield ``(slot_idx, record_bytes)`` for every live record.
 
      Tombstoned slots are skipped.  This is used by the executor's
      heap scanner to iterate all rows in a table page.

      Yields
      ------
      tuple[int, bytes]
          The slot index and the raw record payload.
      """
      for i in range(self.num_slots):
          off, length = self._read_slot(i)
          if off != 0 or length != 0:
              yield i, bytes(self._buf[off:off + length])

OVERFLOW_HEADER = 8  # next_page(u32) + payload_len(u16) + pad(2)
"""Size of the overflow sub-header in bytes.
 
Stored immediately after the standard 24-byte page header::
 
    next_page_id   u32   Next overflow page (INVALID_PAGE if last).
    payload_len    u16   Bytes of payload stored in *this* page.
    _pad           2x    Alignment padding.
 
Packed as ``struct.Struct("<IH2x")``.
"""

def make_overflow_page(page_id: int, payload: bytes, next_page: int = INVALID_PAGE) -> SlottedPage:
  """Create an overflow page storing a raw payload blob.
 
  Overflow pages bypass the slot directory — the payload is written
  directly after the page header + overflow sub-header.

  Parameters
  ----------
  page_id : int
      The page number to assign.
  payload : bytes
      The chunk of data to store.  Must fit in
      ``PAGE_SIZE - HEADER_SIZE - OVERFLOW_HEADER`` bytes (4064).
  next_page : int
      The page id of the next overflow page in the chain, or
      ``INVALID_PAGE`` if this is the last (or only) page.

  Returns
  -------
  SlottedPage
      A page with ``page_type = OVERFLOW``, ready to be written
      to disk via the buffer pool.
  """
  assert len(payload) <= PAGE_SIZE - HEADER_SIZE - OVERFLOW_HEADER
  p = SlottedPage(page_id=page_id, page_type=PageType.OVERFLOW)
  p.overflow_pid = next_page
  # store payload right after header
  start = HEADER_SIZE
  struct.pack_into("<IH2x", p._buf, start, next_page, len(payload))
  p._buf[start + OVERFLOW_HEADER:start + OVERFLOW_HEADER + len(payload)] = payload
  p._write_header()
  return p

def read_overflow_payload(page: SlottedPage) -> tuple[bytes, int]:
    """Extract the payload and chain pointer from an overflow page.
 
    Parameters
    ----------
    page : SlottedPage
        A page with ``page_type == OVERFLOW``.
 
    Returns
    -------
    tuple[bytes, int]
        ``(payload_bytes, next_page_id)``.  The ``next_page_id`` is
        ``INVALID_PAGE`` if this is the last page in the chain.
    """
    start = HEADER_SIZE
    next_pid, plen = struct.unpack_from("<IH", page._buf, start)
    data = bytes(page._buf[start + OVERFLOW_HEADER:start + OVERFLOW_HEADER + plen])
    return data, next_pid
