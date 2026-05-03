"""
Transaction Manager
===================
 
Overview
--------
The transaction manager provides **ACID** guarantees by coordinating
three subsystems:
 
* The **WAL** — ensures durability (the "D" in ACID).  Every
  mutation is logged before any dirty page reaches the data file.
* The **Lock Manager** — ensures isolation (the "I" in ACID).  A
  strict two-phase locking (2PL) protocol prevents concurrent
  transactions from seeing each other's uncommitted changes.
* The **Buffer Pool** — ensures atomicity (the "A" in ACID).  On
  commit, all modified pages are flushed; on abort, the WAL records
  enable undo.
 
Locking model — Strict 2PL
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Every page access must first acquire a lock:
 
* **Shared (S)** — for reads.  Multiple transactions can hold S
  locks on the same page concurrently.
* **Exclusive (X)** — for writes.  Only one transaction can hold an
  X lock, and it blocks all S locks from other transactions.
 
"Strict" means all locks are held **until commit or abort** — they
are never released early.  This prevents cascading aborts and ensures
serialisable isolation.
 
Lock upgrades (S → X) are supported: if a transaction holds S and
no other transaction also holds S on the same page, the lock is
promoted to X without releasing and re-acquiring.
 
Deadlock prevention
~~~~~~~~~~~~~~~~~~~
Instead of detecting deadlock cycles, we use a **timeout** approach:
if a lock request is not granted within 5 seconds, we assume a
deadlock and return ``False`` (the caller aborts the transaction).
This is simple and effective for the moderate concurrency expected
from this engine.
 
Transaction lifecycle
~~~~~~~~~~~~~~~~~~~~~
::
 
    txn = txn_mgr.begin()           # WAL ← BEGIN record
    txn_mgr.acquire(txn, pid, X)    # Lock page
    ...                              # Modify pages
    txn_mgr.commit(txn)             # WAL ← COMMIT, flush pages, release locks
    # or
    txn_mgr.abort(txn)              # WAL ← ABORT, release locks
"""

from __future__ import annotations
 
import enum
import threading
import time
 
from pydb.wal import WAL, LogRecordType
from pydb.cache import BufferPool

class TxnState(enum.Enum):
    """Lifecycle state of a transaction.
 
    Members
    -------
    ACTIVE : str
        The transaction is in progress — it can acquire locks,
        modify pages, and issue SQL statements.
    COMMITTED : str
        The transaction has been committed.  Its effects are durable
        and visible to all subsequent transactions.
    ABORTED : str
        The transaction has been rolled back.  Its locks have been
        released and its modifications should be undone.
    """
    ACTIVE    = "ACTIVE"
    COMMITTED = "COMMITTED"
    ABORTED   = "ABORTED"
    
class LockMode(enum.Enum):
    """Lock type for the two-phase locking protocol.
 
    Members
    -------
    SHARED : str
        Read lock.  Compatible with other ``SHARED`` locks but
        blocks ``EXCLUSIVE`` requests from other transactions.
    EXCLUSIVE : str
        Write lock.  Incompatible with all other locks from other
        transactions.  Required before modifying a page.
    """
    SHARED = "S"
    EXCLUSIVE = "X"
    
class Transaction:
    """In-memory representation of an active transaction.
 
    Created by ``TransactionManager.begin()`` and passed to every
    subsequent operation (lock acquisition, WAL logging, commit/abort).
 
    Attributes
    ----------
    txn_id : int
        Unique, monotonically increasing transaction identifier.
    state : TxnState
        Current lifecycle state.
    locks : dict[int, LockMode]
        Pages currently locked by this transaction and their modes.
        Used for fast lock-already-held checks and for releasing all
        locks on commit/abort.
    modified_pages : set[int]
        Page ids that were written during this transaction.  On
        commit, these pages are flushed from the buffer pool to
        the data file.
    lsn : int
        The LSN of the most recent WAL record produced by this
        transaction.  Stored on modified pages so the recovery
        manager knows which log records have already been applied.
    """
    
    __slots__ = ("txn_id", "state", "locks", "modified_pages", "lsn")
    
    def __init__(self, txn_id: int):
        self.txn_id = txn_id
        self.state = TxnState.ACTIVE
        self.locks: dict[int, LockMode] = {}
        self.modified_pages: set[int] = set()
        self.lsn = 0
    
class LockManager:
    """Page-level S/X lock manager with timeout-based deadlock prevention.
 
    The lock table maps each page id to a lock entry containing:
 
    * ``S`` — set of transaction ids holding shared locks.
    * ``X`` — the transaction id holding the exclusive lock (or ``None``).
    * ``cond`` — a ``threading.Condition`` used to wake blocked
      waiters when a lock is released.
 
    Parameters
    ----------
    timeout : float
        Maximum seconds to wait for a lock before giving up (treated
        as a deadlock).  Defaults to 5.0.
    """
    
    def __init__(self, timeout: float = 5.0):
        self._timeout = timeout
        self._lock = threading.Lock()
        self._table: dict[int, dict] = {}
        
    def _ensure(self, page_id: int) -> dict:
        """Lazily create a lock entry for *page_id* if none exists."""
        if page_id not in self._table:
            self._table[page_id] = {
                "S": set(),
                "X": None,
                "cond": threading.Condition(self._lock),
            }
        return self._table[page_id]
    
    def acquire(self, txn_id: int, page_id: int, mode: LockMode) -> bool:
        """Attempt to acquire a lock, blocking until granted or timed out.
 
        Compatibility matrix::
 
                     Requested
                     S          X
            Held S   ✓ (grant)  ✗ (wait, unless same txn)
            Held X   ✗ (wait)   ✗ (wait, unless same txn)
            None     ✓          ✓
 
        **Lock upgrade**: if *txn_id* already holds S and no other
        transaction holds S, the lock is promoted to X in place.
 
        Parameters
        ----------
        txn_id : int
            The requesting transaction.
        page_id : int
            The page to lock.
        mode : LockMode
            ``SHARED`` or ``EXCLUSIVE``.
 
        Returns
        -------
        bool
            ``True`` if the lock was granted.  ``False`` if the
            timeout expired (probable deadlock — the caller should
            abort the transaction).
        """
        deadline = time.monotonic() + self._timeout
        with self._lock:
            entry = self._ensure(page_id)
            while True:
                if mode == LockMode.SHARED:
                    if entry["X"] is None or entry["X"] == txn_id:
                        entry["S"].add(txn_id)
                        return True
                else: # EXCLUSIVE
                    holders = entry["S"] - {txn_id}
                    if not holders and (entry["X"] is None or entry["X"] == txn_id):
                        entry["X"] = txn_id
                        entry["S"].discard(txn_id)
                        return True
                    if not holders and txn_id in entry["S"] and entry["X"] is None:
                        entry["S"].discard(txn_id)
                        entry["X"] = txn_id
                        return True
                    
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                entry["cond"].wait(timeout=remaining)
                
    def release_all(self, txn_id: int):
        """Release every lock held by *txn_id* and wake blocked waiters.
 
        Called during commit and abort to satisfy the strict-2PL
        requirement that all locks are held until end-of-transaction.
        """
        with self._lock:
            for entry in self._table.values():
                changed = False
                if txn_id in entry["S"]:
                    entry["S"].discard(txn_id)
                    changed = True
                if entry["X"] == txn_id:
                    entry["X"] = None
                    changed = True
                if changed:
                    entry["cond"].notify_all()
                    

class TransactionManager:
    """Coordinates transactions, WAL logging, and the buffer pool.
 
    This is the component that the ``Executor`` talks to for all
    transaction-related operations.  It owns the ``LockManager``
    internally and delegates durability to the ``WAL``.
 
    Parameters
    ----------
    wal : WAL
        The write-ahead log for durability.
    pool : BufferPool
        The buffer pool for flushing dirty pages on commit.
    """
    
    def __init__(self, wal: WAL, pool: BufferPool):
        self._wal = wal
        self._pool = pool
        self._lock_mgr = LockManager()
        self._txn_counter = 0
        self._mu = threading.Lock()
        self._active: dict[int, Transaction] = {}
        
    def begin(self) -> Transaction:
        """Start a new transaction.
 
        Assigns a unique transaction id, writes a ``BEGIN`` record
        to the WAL, and registers the transaction as active.
 
        Returns
        -------
        Transaction
            A fresh ``Transaction`` object in ``ACTIVE`` state.
        """
        with self._mu:
            self._txn_counter += 1
            txn = Transaction(self._txn_counter)
        txn.lsn = self._wal.append(txn.txn_id, LogRecordType.BEGIN)
        with self._mu:
            self._active[txn.txn_id] = txn
        return txn
    
    def acquire(self, txn: Transaction, page_id: int, mode: LockMode) -> bool:
        """Acquire a lock on behalf of a transaction.
 
        Short-circuits if the transaction already holds an equal or
        stronger lock on the page (e.g. asking for S when X is held).
 
        Parameters
        ----------
        txn : Transaction
            The requesting transaction.
        page_id : int
            The page to lock.
        mode : LockMode
            ``SHARED`` or ``EXCLUSIVE``.
 
        Returns
        -------
        bool
            ``True`` if granted, ``False`` on timeout (abort the txn).
        """
        if page_id in txn.locks:
            held = txn.locks[page_id]
            if held == LockMode.EXCLUSIVE or mode == LockMode.SHARED:
                return True
            
        ok = self._lock_mgr.acquire(txn.txn_id, page_id, mode)
        if ok:
            txn.locks[page_id] = mode
        return ok
    
    def log_update(self, txn: Transaction, page_id: int, before: bytes = b"", after: bytes = b"") -> int:
        """Write an ``UPDATE`` log record to the WAL.
 
        The payload stores both the before-image and after-image of
        the modified region, each prefixed with a 4-byte length.
        During recovery, redo replays the after-image; undo restores
        the before-image.
 
        Parameters
        ----------
        txn : Transaction
            The modifying transaction.
        page_id : int
            The data-file page being modified.
        before : bytes
            The page content before the modification.
        after : bytes
            The page content after the modification.
 
        Returns
        -------
        int
            The LSN of the new log record.
        """
        data = (len(before).to_bytes(4, "little") + before +
                len(after).to_bytes(4, "little") + after)
        lsn = self._wal.append(txn.txn_id, LogRecordType.UPDATE, page_id, data)
        txn.modified_pages.add(page_id)
        txn.lsn = lsn
        return lsn
    
    def commit(self, txn: Transaction):
        """Commit a transaction, making its effects durable.
 
        Steps:
 
        1. Write a ``COMMIT`` record to the WAL.
        2. Flush the WAL to stable storage.
        3. Flush all pages modified by this transaction from the
           buffer pool to the data file.
        4. Release all locks held by this transaction.
        5. Mark the transaction as ``COMMITTED``.
        """
        self._wal.append(txn.txn_id, LogRecordType.COMMIT)
        self._wal.flush()
        
        for pid in txn.modified_pages:
            self._pool.flush_page(pid)
        self._lock_mgr.release_all(txn.txn_id)
        txn.state = TxnState.COMMITTED
        with self._mu:
            self._active.pop(txn.txn_id, None)
            
    def abort(self, txn: Transaction):
        """Abort a transaction, discarding its effects.
 
        Writes an ``ABORT`` record to the WAL and releases all locks.
        Modified pages remain in the buffer pool but their changes
        are logically undone — the WAL's before-images can restore
        the original state during crash recovery if needed.
        """
        self._wal.append(txn.txn_id, LogRecordType.ABORT)
        self._lock_mgr.release_all(txn.txn_id)
        txn.state = TxnState.ABORTED
        with self._mu:
            self._active.pop(txn.txn_id, None)
            
    @property
    def auto_txn(self) -> Transaction:
        """Begin and return an implicit auto-commit transaction.
 
        Used by the executor for single-statement queries that are
        not wrapped in an explicit ``BEGIN`` / ``COMMIT`` block.
        """
        return self.begin()
