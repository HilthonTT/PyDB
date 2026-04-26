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
