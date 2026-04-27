"""
B+Tree Index
============
Disk-backed B+Tree with:
 - Variable-length keys (serialised as length-prefixed bytes)
 - Leaf pages linked for range scans
 - Sophisticated balancing: split, merge, key redistribution
 - Bulk-loading path
 
Node layout (encoded inside a SlottedPage):
  Internal node: [num_keys(u16)] [child0(u32)] {key_i, child_i+1} ...
  Leaf node:     [num_keys(u16)] [next_leaf(u32)] {key_i, rid_i} ...
 
Keys are compared as raw bytes (lexicographic).  The caller is
responsible for encoding typed values into comparable byte strings
(see record.py encode_key / decode_key helpers).
"""

from __future__ import annotations
 
import struct
from typing import Optional
 
from pydb import INVALID_PAGE, BTREE_ORDER, PAGE_SIZE, HEADER_SIZE
from pydb.page import SlottedPage, PageType, RID
from pydb.cache import BufferPool

# Maximum keys per node — we split at ORDER and target ORDER//2 minimum.
ORDER = BTREE_ORDER
MIN_KEYS = ORDER // 2

def _encode_key(key: bytes) -> bytes:
    return struct.pack("<H", len(key)) + key

def _decode_key(buf: bytes, pos: int) -> tuple[bytes, int]:
    klen = struct.unpack_from("<H", buf, pos)[0]
    pos += 2
    return buf[pos:pos + klen], pos + klen

def _serialize_internal(children: list[int], keys: list[bytes]) -> bytes:
  """Pack an internal node: num_keys | child0 | (key, child)*

  Args:
      children (list[int]): The children to pack.
      keys (list[bytes]): The keys.

  Returns:
      bytes: The packed value.
  """
  parts = [struct.pack("<HI", len(keys), children[0])]
  for i, k in enumerate(keys):
    parts.append(_encode_key(k))
    parts.append(struct.pack("<I", children[i + 1]))
  return b"".join(parts)

def _deserialize_internal(data: bytes) -> tuple[list[bytes], list[int]]:
  pos = 0
  nk = struct.unpack_from("<H", data, pos)[0]; pos += 2 # H is an unsigned short -> 2 bytes
  c0 = struct.unpack("<I", data, pos)[0]; pos += 4 # I is an unsigned integer -> 4 bytes
  
  keys: list[bytes] = []
  children: list[int] = [c0]
  
  for _ in range(nk):
    k, pos = _decode_key(data, pos)
    keys.append(k)
    c = struct.unpack_from("<I", data, pos)[0]; pos += 4
    children.append(c)
  return keys, children

def _serialize_leaf(keys: list[bytes], rids: list[RID], next_leaf: int) -> bytes:
    parts = [struct.pack("<HI", len(keys), next_leaf)]
    for k, r in zip(keys, rids):
        parts.append(_encode_key(k))
        parts.append(r.to_bytes())
    return b"".join(parts)
  
def _deserialize_leaf(data: bytes) -> tuple[list[bytes], list[RID], int]:
  pos = 0
  nk = struct.unpack_from("<H", data, pos)[0]; pos += 2
  next_leaf = struct.unpack_from("<I", data, pos)[0]; pos += 4
  keys: list[bytes] = []
  rids: list[RID] = []
  
  for _ in range(nk):
    k, pos = _decode_key(data, pos)
    keys.append(k)
    r = RID.from_bytes(data[pos:pos + 6]); pos += 6
    rids.append(r)
  return keys, rids, next_leaf

class BPlusTree:
  """
  Disk-resident B+Tree stored across SlottedPages managed by a BufferPool.
  
  Each tree is identified by its *root_page_id* (persisted in the catalog).
  """
  
  def __init__(self, pool: BufferPool, root_page_id: int = INVALID_PAGE):
    self._pool = pool
    self.root_pid = root_page_id
    
  def _alloc_node(self, ptype: PageType) -> SlottedPage:
      p = self._pool.new_page()
      p.page_type = ptype
      return p
    
  def _load(self, pid: int) -> SlottedPage:
    return self._pool.fetch_page(pid)
  
  def _release(self, pid: int, dirty: bool = False):
    self._pool.unpin(pid, dirty)
    
  def _write_node(self, page: SlottedPage, data: bytes):
    """Store serialised node as slot-0 of the page."""
    # clear and rewrite
    page.num_slots = 0
    page.free_offset = HEADER_SIZE
    page.free_end = PAGE_SIZE
    page._write_header()
    page.insert(data)
    
  def _read_node(self, page: SlottedPage) -> bytes:
    return page.read(0)
  
  def create(self) -> int:
    """Creates an empty tree and returns its root page id"""
    leaf = self._alloc_node(PageType.BTREE_LEAF)
    data = _serialize_leaf([], [], INVALID_PAGE)
    self._write_node(leaf, data)
    self.root_pid = leaf.page_id
    self._release(leaf.page_id, dirty=True)
    return self.root_pid
  
  def search(self, key: bytes) -> Optional[RID]:
    """Searches the tree with the provided key to find the row identifier"""
    if self.root_pid == INVALID_PAGE:
      return None
    
    pid = self._find_leaf(key)
    page = self._load(pid)
    keys, rids, _ = _deserialize_leaf(self._read_node(page))
    self._release(pid)
    for i, k in enumerate(keys):
      if k == key:
        return rids[i]
    return None
  
  def _find_leaf(self, keys: bytes) -> int:
    pid = self.root_pid
    
    while True:
      page = self._load(pid)
      if page.page_type == PageType.BTREE_LEAF:
        self._release(pid)
        return pid
      
      keys, children = _deserialize_internal(self._read_node(page))
      self._release(pid)
      # binary search for child
      idx = 0
      for i, k in enumerate(keys):
        if keys > k:
          idx = i + 1
        else:
          break
        
      pid = children[idx]
      return pid
    
  def insert(self, key: bytes, rid: RID):
    if self.root_pid == INVALID_PAGE:
        self.create()

    split = self._insert_rec(self.root_pid, key, rid)
    if split is not None:
        median, new_pid = split
        new_root = self._alloc_node(PageType.BTREE_INTERNAL)
        data = _serialize_internal([self.root_pid, new_pid], [median])
        self._write_node(new_root, data)
        self._release(new_root.page_id, dirty=True)
        self.root_pid = new_root.page_id 
        
  def _insert_rec(self, pid: int, key: bytes, rid: RID) -> Optional[tuple[bytes, int]]:
    page = self._load(pid)
    
    if page.page_type == PageType.BTREE_LEAF:
      keys, rids, next_leaf = _deserialize_leaf(self._read_node(page))
      # find insert position (duplicate keys OK - multi-valued index)
      pos = 0
      for i, k in enumerate(keys):
        if key >= k:
          pos = i + 1
        else:
          break
        
      keys.insert(pos, key)
      rids.insert(pos, rid)
      
      if len(keys) <= ORDER:
        self._write_node(page, _serialize_leaf(keys, rids, next_leaf))
        self._release(pid, dirty=True)
        return None
      
      # split
      mid = len(keys) // 2
      right_page = self._alloc_node(PageType.BTREE_LEAF)
      r_keys, r_rids = keys[mid:], rids[mid:]
      l_keys, l_rids = keys[:mid], rids[:mid]
      
      right_pid = right_page.page_id
      self._write_node(right_page, _serialize_leaf(r_keys, r_rids, next_leaf))
      self._write_node(page, _serialize_leaf(l_keys, l_rids, right_pid))
      self._release(right_pid, dirty=True)
      self._release(pid, dirty=True)
      return (r_keys[0], right_pid)
    
    # internal node
    keys, children = _deserialize_internal(self._read_node(page))
    self._release(pid)

    idx = 0
    for i, k in enumerate(keys):
        if key >= k:
            idx = i + 1
        else:
            break

    split = self._insert_rec(children[idx], key, rid)
    if split is None:
        return None

    median, new_child = split
    page = self._load(pid)
    keys, children = _deserialize_internal(self._read_node(page))
    keys.insert(idx, median)
    children.insert(idx + 1, new_child)

    if len(keys) <= ORDER:
        self._write_node(page, _serialize_internal(children, keys))
        self._release(pid, dirty=True)
        return None

    mid = len(keys) // 2
    up_key = keys[mid]
    r_keys = keys[mid + 1:]
    r_children = children[mid + 1:]
    l_keys = keys[:mid]
    l_children = children[:mid + 1]

    right = self._alloc_node(PageType.BTREE_INTERNAL)
    self._write_node(right, _serialize_internal(r_children, r_keys))
    self._write_node(page, _serialize_internal(l_children, l_keys))
    rpid = right.page_id
    self._release(rpid, dirty=True)
    self._release(pid, dirty=True)
    return (up_key, rpid)
  
  def delete(self, key: bytes, rid: Optional[RID] = None) -> bool:
    """Delete key (optionally matching a specific RID). Returns True if found."""
    if self.root_pid == INVALID_PAGE:
        return False
    deleted, underflow = self._delete_rec(self.root_pid, key, rid)
    if deleted and underflow:
        # check if root is internal with one child
        page = self._load(self.root_pid)
        if page.page_type == PageType.BTREE_INTERNAL:
            keys, children = _deserialize_internal(self._read_node(page))
            if len(keys) == 0:
                old_root = self.root_pid
                self.root_pid = children[0]
                self._release(old_root)
                self._pool.delete_page(old_root)
                return True
        self._release(self.root_pid)
    return deleted
 
  def _delete_rec(self, pid: int, key: bytes, rid: Optional[RID]) -> tuple[bool, bool]:
        page = self._load(pid)
 
        if page.page_type == PageType.BTREE_LEAF:
            keys, rids, next_leaf = _deserialize_leaf(self._read_node(page))
            found_idx = -1
            for i, k in enumerate(keys):
                if k == key and (rid is None or rids[i] == rid):
                    found_idx = i
                    break
            if found_idx == -1:
                self._release(pid)
                return False, False
 
            keys.pop(found_idx)
            rids.pop(found_idx)
            self._write_node(page, _serialize_leaf(keys, rids, next_leaf))
            self._release(pid, dirty=True)
            return True, len(keys) < MIN_KEYS
 
        # internal node
        keys, children = _deserialize_internal(self._read_node(page))
        self._release(pid)
 
        idx = 0
        for i, k in enumerate(keys):
            if key >= k:
                idx = i + 1
            else:
                break
 
        deleted, underflow = self._delete_rec(children[idx], key, rid)
        if not deleted:
            return False, False
        if not underflow:
            return True, False
 
        # handle underflow via redistribute or merge
        return True, self._handle_underflow(pid, idx)
 
  def _handle_underflow(self, parent_pid: int, child_idx: int) -> bool:
      """Try to redistribute with a sibling; if not possible, merge. Returns True if parent underflowed."""
      parent = self._load(parent_pid)
      keys, children = _deserialize_internal(self._read_node(parent))

      child = self._load(children[child_idx])
      is_leaf = child.page_type == PageType.BTREE_LEAF

      # try left sibling
      if child_idx > 0:
          left = self._load(children[child_idx - 1])
          if is_leaf:
              lk, lr, ln = _deserialize_leaf(self._read_node(left))
              ck, cr, cn = _deserialize_leaf(self._read_node(child))
              if len(lk) > MIN_KEYS:
                  # redistribute: move last from left to child
                  ck.insert(0, lk.pop())
                  cr.insert(0, lr.pop())
                  keys[child_idx - 1] = ck[0]
                  self._write_node(left, _serialize_leaf(lk, lr, ln))
                  self._write_node(child, _serialize_leaf(ck, cr, cn))
                  self._write_node(parent, _serialize_internal(children, keys))
                  self._release(left.page_id, True)
                  self._release(child.page_id, True)
                  self._release(parent_pid, True)
                  return False
              # merge into left
              lk.extend(ck)
              lr.extend(cr)
              self._write_node(left, _serialize_leaf(lk, lr, cn))
              self._release(left.page_id, True)
              self._release(child.page_id)
              self._pool.delete_page(children[child_idx])
              keys.pop(child_idx - 1)
              children.pop(child_idx)
              self._write_node(parent, _serialize_internal(children, keys))
              self._release(parent_pid, True)
              return len(keys) < MIN_KEYS
          else:
              # internal node redistribute / merge
              lk, lc = _deserialize_internal(self._read_node(left))
              ck, cc = _deserialize_internal(self._read_node(child))
              if len(lk) > MIN_KEYS:
                  sep = keys[child_idx - 1]
                  ck.insert(0, sep)
                  cc.insert(0, lc.pop())
                  keys[child_idx - 1] = lk.pop()
                  self._write_node(left, _serialize_internal(lc, lk))
                  self._write_node(child, _serialize_internal(cc, ck))
                  self._write_node(parent, _serialize_internal(children, keys))
                  self._release(left.page_id, True)
                  self._release(child.page_id, True)
                  self._release(parent_pid, True)
                  return False
              sep = keys[child_idx - 1]
              lk.append(sep)
              lk.extend(ck)
              lc.extend(cc)
              self._write_node(left, _serialize_internal(lc, lk))
              self._release(left.page_id, True)
              self._release(child.page_id)
              self._pool.delete_page(children[child_idx])
              keys.pop(child_idx - 1)
              children.pop(child_idx)
              self._write_node(parent, _serialize_internal(children, keys))
              self._release(parent_pid, True)
              return len(keys) < MIN_KEYS

      # try right sibling
      if child_idx < len(children) - 1:
          right = self._load(children[child_idx + 1])
          if is_leaf:
              ck, cr, cn = _deserialize_leaf(self._read_node(child))
              rk, rr, rn = _deserialize_leaf(self._read_node(right))
              if len(rk) > MIN_KEYS:
                  ck.append(rk.pop(0))
                  cr.append(rr.pop(0))
                  keys[child_idx] = rk[0]
                  self._write_node(child, _serialize_leaf(ck, cr, right.page_id))
                  self._write_node(right, _serialize_leaf(rk, rr, rn))
                  self._write_node(parent, _serialize_internal(children, keys))
                  self._release(right.page_id, True)
                  self._release(child.page_id, True)
                  self._release(parent_pid, True)
                  return False
              ck.extend(rk)
              cr.extend(rr)
              self._write_node(child, _serialize_leaf(ck, cr, rn))
              self._release(child.page_id, True)
              self._release(right.page_id)
              self._pool.delete_page(children[child_idx + 1])
              keys.pop(child_idx)
              children.pop(child_idx + 1)
              self._write_node(parent, _serialize_internal(children, keys))
              self._release(parent_pid, True)
              return len(keys) < MIN_KEYS
          else:
              ck, cc = _deserialize_internal(self._read_node(child))
              rk, rc = _deserialize_internal(self._read_node(right))
              if len(rk) > MIN_KEYS:
                  sep = keys[child_idx]
                  ck.append(sep)
                  cc.append(rc.pop(0))
                  keys[child_idx] = rk.pop(0)
                  self._write_node(child, _serialize_internal(cc, ck))
                  self._write_node(right, _serialize_internal(rc, rk))
                  self._write_node(parent, _serialize_internal(children, keys))
                  self._release(right.page_id, True)
                  self._release(child.page_id, True)
                  self._release(parent_pid, True)
                  return False
              sep = keys[child_idx]
              ck.append(sep)
              ck.extend(rk)
              cc.extend(rc)
              self._write_node(child, _serialize_internal(cc, ck))
              self._release(child.page_id, True)
              self._release(right.page_id)
              self._pool.delete_page(children[child_idx + 1])
              keys.pop(child_idx)
              children.pop(child_idx + 1)
              self._write_node(parent, _serialize_internal(children, keys))
              self._release(parent_pid, True)
              return len(keys) < MIN_KEYS

      self._release(child.page_id)
      self._release(parent_pid)
      return False
    