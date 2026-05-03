"""
B+Tree Index
============

Overview
--------
The B+Tree is the standard index structure for relational databases.
It provides O(log N) point lookups, O(log N + M) range scans (where
M is the number of matching keys), and O(log N) insertions and
deletions — all with excellent disk locality because each node is
stored in a single page.

This implementation is **disk-resident**: every node lives inside a
``SlottedPage`` managed by the ``BufferPool``.  The tree never
materialises the full structure in memory — it loads nodes on demand,
performs the operation, and unpins them.

B+Tree vs B-Tree
~~~~~~~~~~~~~~~~~
In a B+Tree (as opposed to a plain B-Tree), **all data lives in the
leaf nodes**.  Internal nodes store only keys and child pointers, acting
as a multi-level routing directory.  This has two advantages:

1. **Internal nodes are denser** — they pack more keys per page,
   keeping the tree shallower.
2. **Leaf nodes are linked** — a ``next_leaf`` pointer chains all
   leaves in key order, so range scans walk a flat linked list
   instead of bouncing up and down the tree.

Node layout
~~~~~~~~~~~
Nodes are serialised into a byte blob and stored as **slot 0** of
their ``SlottedPage``.  Two formats exist:

**Internal node** — ``[num_keys(u16)][child0(u32)]{key_i, child_i+1}*``

An internal node with *N* keys has *N + 1* child pointers.  ``child0``
comes before the first key; subsequent children interleave with keys::

    child0 | key0 | child1 | key1 | child2 | ... | keyN-1 | childN

To route a search key *K*, we find the rightmost ``key_i`` where
``K >= key_i`` and descend into ``child_{i+1}`` (or ``child0`` if *K*
is less than every key).

**Leaf node** — ``[num_keys(u16)][next_leaf(u32)]{key_i, rid_i}*``

Each entry pairs a key with a ``RID`` (page_id + slot_idx) pointing
to the actual row in the heap.  ``next_leaf`` is the page id of the
next leaf in key order (``INVALID_PAGE`` if last).

Key encoding
~~~~~~~~~~~~
Keys are compared as **raw bytes** (lexicographic order).  The
``catalog.encode_key`` function transforms typed column values
(integers, floats, text) into byte strings that sort correctly —
for example, signed integers are encoded with a sign-bit flip so
that negative values sort before positive ones.

Within the serialised node, each key is length-prefixed:
``[key_len(u16)][key_bytes]``.

Balancing
~~~~~~~~~
The tree maintains the B+Tree invariant that every non-root node
holds between ``MIN_KEYS`` (``ORDER // 2 = 64``) and ``ORDER``
(128) keys:

* **Split** — when a leaf or internal node exceeds ``ORDER`` keys
  after an insertion, it is split into two nodes and the median
  key is pushed up to the parent.
* **Merge** — when a node drops below ``MIN_KEYS`` after a deletion,
  it is merged with a sibling, pulling the separator key down from
  the parent.
* **Redistribute** — before merging, we check whether a sibling
  has more than ``MIN_KEYS`` entries.  If so, we rotate one key
  through the parent instead of merging.  This keeps both nodes
  alive and avoids cascading underflows.

If the root splits, a new root is created with two children.  If
the root's only child merges away, the tree shrinks by one level.
"""

from __future__ import annotations

import struct
from typing import Optional

from pydb import INVALID_PAGE, BTREE_ORDER, PAGE_SIZE, HEADER_SIZE
from pydb.page import SlottedPage, PageType, RID
from pydb.cache import BufferPool
from pydb.txn import Transaction, TransactionManager

ORDER = BTREE_ORDER
"""Maximum keys per B+Tree node.  A node splits when it exceeds this."""

MIN_KEYS = ORDER // 2
"""Minimum keys per non-root B+Tree node.  A node underflows below this."""


# Key serialisation

def _encode_key(key: bytes) -> bytes:
    """Length-prefix a key: ``[u16 length][raw bytes]``.

    This is the on-disk format used *inside* serialised node blobs,
    not to be confused with ``catalog.encode_key`` which transforms
    typed values into sort-comparable byte strings.
    """
    return struct.pack("<H", len(key)) + key


def _decode_key(buf: bytes, pos: int) -> tuple[bytes, int]:
    """Read a length-prefixed key starting at *pos*.

    Returns ``(key_bytes, new_pos)`` where *new_pos* is the byte
    offset immediately after the key.
    """
    klen = struct.unpack_from("<H", buf, pos)[0]
    pos += 2
    return buf[pos:pos + klen], pos + klen


# Node serialisation 

def _serialize_internal(children: list[int], keys: list[bytes]) -> bytes:
    """Pack an internal node into bytes.

    Format: ``num_keys(u16) | child0(u32) | {key_i | child_{i+1}}*``

    Parameters
    ----------
    children : list[int]
        Child page ids.  Always ``len(keys) + 1`` entries.
    keys : list[bytes]
        Separator keys.  Each key is length-prefixed in the output.

    Returns
    -------
    bytes
        The serialised blob, stored as slot 0 of the node's page.
    """
    parts = [struct.pack("<HI", len(keys), children[0])]
    for i, k in enumerate(keys):
        parts.append(_encode_key(k))
        parts.append(struct.pack("<I", children[i + 1]))
    return b"".join(parts)


def _deserialize_internal(data: bytes) -> tuple[list[bytes], list[int]]:
    """Unpack an internal node from bytes.

    Returns
    -------
    tuple[list[bytes], list[int]]
        ``(keys, children)`` where ``len(children) == len(keys) + 1``.
    """
    pos = 0
    nk = struct.unpack_from("<H", data, pos)[0]; pos += 2
    c0 = struct.unpack_from("<I", data, pos)[0]; pos += 4
    keys: list[bytes] = []
    children: list[int] = [c0]
    for _ in range(nk):
        k, pos = _decode_key(data, pos)
        keys.append(k)
        c = struct.unpack_from("<I", data, pos)[0]; pos += 4
        children.append(c)
    return keys, children


def _serialize_leaf(keys: list[bytes], rids: list[RID], next_leaf: int) -> bytes:
    """Pack a leaf node into bytes.

    Format: ``num_keys(u16) | next_leaf(u32) | {key_i | rid_i}*``

    Parameters
    ----------
    keys : list[bytes]
        Index keys (length-prefixed in the output).
    rids : list[RID]
        Corresponding row identifiers (6 bytes each).
    next_leaf : int
        Page id of the next leaf in key order, or ``INVALID_PAGE``.
    """
    parts = [struct.pack("<HI", len(keys), next_leaf)]
    for k, r in zip(keys, rids):
        parts.append(_encode_key(k))
        parts.append(r.to_bytes())
    return b"".join(parts)


def _deserialize_leaf(data: bytes) -> tuple[list[bytes], list[RID], int]:
    """Unpack a leaf node from bytes.

    Returns
    -------
    tuple[list[bytes], list[RID], int]
        ``(keys, rids, next_leaf_page_id)``.
    """
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
    """Disk-resident B+Tree stored across ``SlottedPage`` objects.

    Each tree instance is identified by its **root page id**, which
    is persisted in the ``IndexDef`` inside the catalog.  The tree
    does not cache any node data itself — every access goes through
    the ``BufferPool``, so the LRU-K replacement policy automatically
    keeps hot nodes in memory.

    Parameters
    ----------
    pool : BufferPool
        The buffer pool used for all page I/O.
    root_page_id : int
        The page id of the current root node.  ``INVALID_PAGE`` if
        the tree has not been created yet.

    Attributes
    ----------
    root_pid : int
        The current root page id.  This may change after insertions
        (root split) or deletions (root collapse).  The caller is
        responsible for persisting the new value in the catalog.
    """

    def __init__(self, pool: BufferPool, root_page_id: int = INVALID_PAGE,
                 txn: Optional[Transaction] = None, 
                 txn_mgr: Optional[TransactionManager] = None):
        self._pool = pool
        self.root_pid = root_page_id
        self._txn = txn
        self._txn_mgr = txn_mgr

    def _log_mutation(self, page: SlottedPage, before: bytes):
        """Log a page mutation to the WAL if a transaction context is available."""
        if self._txn and self._txn_mgr:
            after = page.to_bytes()
            lsn = self._txn_mgr.log_update(self._txn, page.page_id, before, after)
            page.lsn = lsn
            page._write_header()

    def _alloc_node(self, ptype: PageType) -> SlottedPage:
        """Allocate a fresh page from the buffer pool and set its type."""
        p = self._pool.new_page()
        p.page_type = ptype
        return p

    def _load(self, pid: int) -> SlottedPage:
        """Fetch and pin a node page from the buffer pool."""
        return self._pool.fetch_page(pid)

    def _release(self, pid: int, dirty: bool = False):
        """Unpin a node page, optionally marking it dirty."""
        self._pool.unpin(pid, dirty)

    def _write_node(self, page: SlottedPage, data: bytes):
        """Replace the node's serialised content (slot 0) with *data*.

        Clears the page back to an empty state, then inserts the
        new blob.  This is safe because B+Tree nodes always occupy
        exactly one slot per page.
        """
        before = page.to_bytes()
        page.num_slots = 0
        page.free_offset = HEADER_SIZE
        page.free_end = PAGE_SIZE
        page._write_header()
        page.insert(data)
        self._log_mutation(page, before)

    def _read_node(self, page: SlottedPage) -> bytes:
        """Read the serialised node blob from slot 0 of *page*."""
        return page.read(0)
      
    def create(self) -> int:
        """Create an empty B+Tree (a single empty leaf page).

        Returns
        -------
        int
            The root page id of the newly created tree.
        """
        leaf = self._alloc_node(PageType.BTREE_LEAF)
        data = _serialize_leaf([], [], INVALID_PAGE)
        self._write_node(leaf, data)
        self.root_pid = leaf.page_id
        self._release(leaf.page_id, dirty=True)
        return self.root_pid

    def search(self, key: bytes) -> Optional[RID]:
        """Point lookup: find the ``RID`` associated with *key*.

        Descends from the root to the correct leaf, then does a
        linear scan of the leaf's keys.

        Parameters
        ----------
        key : bytes
            The search key (encoded via ``catalog.encode_key``).

        Returns
        -------
        RID or None
            The row identifier if found, or ``None``.
        """
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

    def range_scan(self, lo: Optional[bytes] = None, hi: Optional[bytes] = None):
        """Yield ``(key, RID)`` pairs for all keys in ``[lo, hi]``.

        Finds the leaf containing *lo* (or the first leaf if *lo* is
        ``None``), then walks the leaf chain via ``next_leaf`` pointers
        until a key exceeds *hi* (or the chain ends).

        Parameters
        ----------
        lo : bytes or None
            Inclusive lower bound.  ``None`` means scan from the start.
        hi : bytes or None
            Inclusive upper bound.  ``None`` means scan to the end.

        Yields
        ------
        tuple[bytes, RID]
            Each matching key and its row identifier.
        """
        if self.root_pid == INVALID_PAGE:
            return
        pid = self._find_leaf(lo if lo else b"")
        while pid != INVALID_PAGE:
            page = self._load(pid)
            keys, rids, next_leaf = _deserialize_leaf(self._read_node(page))
            self._release(pid)
            for k, r in zip(keys, rids):
                if lo and k < lo:
                    continue
                if hi and k > hi:
                    return
                yield k, r
            pid = next_leaf

    def _find_leaf(self, key: bytes) -> int:
        """Descend from the root to the leaf that should contain *key*.

        At each internal node, finds the child whose key range
        includes *key* by scanning the separator keys left to right.

        Returns
        -------
        int
            The page id of the target leaf.
        """
        pid = self.root_pid
        while True:
            page = self._load(pid)
            if page.page_type == PageType.BTREE_LEAF:
                self._release(pid)
                return pid
            keys, children = _deserialize_internal(self._read_node(page))
            self._release(pid)
            idx = 0
            for i, k in enumerate(keys):
                if key >= k:
                    idx = i + 1
                else:
                    break
            pid = children[idx]

    def insert(self, key: bytes, rid: RID):
        """Insert a ``(key, RID)`` pair into the tree.

        Duplicate keys are allowed — this supports non-unique indexes
        where multiple rows share the same indexed value.

        If the insertion causes the root to split, a new root is
        created with the two halves as children, increasing the
        tree's height by one.

        Parameters
        ----------
        key : bytes
            The index key (encoded via ``catalog.encode_key``).
        rid : RID
            The row identifier pointing to the heap record.
        """
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
        """Recursive insert.  Returns ``None`` if no split, or
        ``(median_key, new_right_page_id)`` if the node split.

        **Leaf split**: the leaf's keys are divided at the midpoint.
        The right half moves to a new leaf page, and the first key
        of the right half is promoted to the parent as the separator.

        **Internal split**: similar, but the median key is *removed*
        from both halves (pushed up) rather than duplicated.
        """
        page = self._load(pid)

        if page.page_type == PageType.BTREE_LEAF:
            keys, rids, next_leaf = _deserialize_leaf(self._read_node(page))
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

            # Split at midpoint.
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

        # Internal node — recurse into the correct child.
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

        # A child split — insert the promoted key + new child pointer.
        median, new_child = split
        page = self._load(pid)
        keys, children = _deserialize_internal(self._read_node(page))
        keys.insert(idx, median)
        children.insert(idx + 1, new_child)

        if len(keys) <= ORDER:
            self._write_node(page, _serialize_internal(children, keys))
            self._release(pid, dirty=True)
            return None

        # Internal node split.
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
        """Delete a key from the tree.

        If *rid* is provided, only the entry matching both key and
        RID is removed (necessary for non-unique indexes where the
        same key maps to multiple rows).  If *rid* is ``None``, the
        first matching key is removed.

        After deletion, if the root is an internal node with zero
        keys (one child), the tree shrinks: the sole child becomes
        the new root.

        Parameters
        ----------
        key : bytes
            The key to delete.
        rid : RID or None
            Optional row identifier to disambiguate duplicates.

        Returns
        -------
        bool
            ``True`` if a matching entry was found and removed.
        """
        if self.root_pid == INVALID_PAGE:
            return False
        deleted, underflow = self._delete_rec(self.root_pid, key, rid)
        if deleted and underflow:
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
        """Recursive delete.  Returns ``(deleted, underflow)``.

        *deleted* is ``True`` if the key was found and removed.
        *underflow* is ``True`` if the node now has fewer than
        ``MIN_KEYS`` entries and the parent needs to rebalance.
        """
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

        # Internal node — recurse into the correct child.
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

        return True, self._handle_underflow(pid, idx)

    def _handle_underflow(self, parent_pid: int, child_idx: int) -> bool:
        """Rebalance after a child underflows (fewer than ``MIN_KEYS``).

        Strategy (tried in order):

        1. **Redistribute from left sibling** — if the left sibling
           has more than ``MIN_KEYS``, rotate its last key into the
           child through the parent's separator.
        2. **Merge with left sibling** — combine the child into the
           left sibling and pull the separator down from the parent.
        3. **Redistribute from right sibling** — symmetric to (1).
        4. **Merge with right sibling** — symmetric to (2).

        For **leaf** nodes, redistribution moves a key-RID pair
        directly and updates the parent separator to the new first
        key of the right node.

        For **internal** nodes, redistribution rotates through the
        parent separator: the sibling's key becomes the new separator,
        and the old separator moves into the child.

        Returns
        -------
        bool
            ``True`` if the parent itself now underflows (merge
            cascaded upward), requiring further rebalancing.
        """
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
                    # Redistribute: move last entry from left → child.
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
                # Merge child into left sibling.
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
      