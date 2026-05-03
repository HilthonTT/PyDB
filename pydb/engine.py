"""
Database Engine
===============
 
Overview
--------
The ``Database`` class is the top-level **façade** that wires every
subsystem together into a coherent engine.  It is the only class that
external code needs to interact with — callers pass in a SQL string
and get back a result dictionary.
 
Component wiring
~~~~~~~~~~~~~~~~
On construction, ``Database`` creates (or reopens) the following
components in bottom-up order::
 
    DiskManager  →  data file I/O
         ↓
    BufferPool   →  LRU-K page cache
         ↓
    WAL          →  crash-recovery log
         ↓
    TransactionManager  →  locking + durability
         ↓
    Catalog      →  table / index metadata
         ↓
    Planner      →  AST → physical plan
         ↓
    Executor     →  plan → results
 
The ``execute(sql)`` method chains:
``parse_sql`` → ``Planner.plan`` → ``Executor.execute``, catching
errors at each stage and returning them as messages in the result dict.
 
Persistence
~~~~~~~~~~~
All data pages live in a single file (``data.db``) managed by the
``DiskManager``.  The WAL lives in ``wal.log``.  The catalog is
stored as ``catalog.json`` (JSON, not binary) for easy debugging.
 
On ``close()``, the buffer pool is flushed (all dirty pages written
to ``data.db``), the catalog is serialised to ``catalog.json``, and
the WAL file handle is closed.
 
Usage
~~~~~
::
 
    db = Database("mydb")
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    db.execute("INSERT INTO users (name) VALUES ('Alice')")
    result = db.execute("SELECT * FROM users")
    # result == {"columns": ["id","name"], "rows": [[1,"Alice"]], "message": "1 row(s)"}
    db.close()
"""

from __future__ import annotations
 
from pathlib import Path
 
from pydb import BUFFER_POOL_CAP
from pydb.storage import DiskManager
from pydb.cache import BufferPool
from pydb.wal import WAL, LogRecordType
from pydb.txn import TransactionManager, _parse_update_payload
from pydb.catalog import Catalog
from pydb.page import SlottedPage
from pydb.parser import parse_sql, ParseError
from pydb.planner import Planner
from pydb.executor import Executor, ExecutionError
from pydb.auth import UserStore

class Database:
    """Self-contained database engine.
 
    All persistent state lives under a single directory containing
    ``data.db`` (pages), ``wal.log`` (write-ahead log), and
    ``catalog.json`` (table/index metadata).
 
    Parameters
    ----------
    path : str
        Path to the database directory.  Created automatically if
        it doesn't exist.
    buffer_pool_size : int
        Number of page frames in the buffer pool.  Defaults to
        ``BUFFER_POOL_CAP`` (1024 = 4 MiB with 4 KiB pages).
    """
    
    def __init__(self, path: str = "pydb_data", buffer_pool_size: int = BUFFER_POOL_CAP):
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        
        self._disk = DiskManager(self._dir / "data.db")
        self._wal = WAL(self._dir / "wal.log")
        self._recover()
        self._pool = BufferPool(self._disk, capacity=buffer_pool_size, wal=self._wal)
        self._txn = TransactionManager(self._wal, self._pool)
        
        cat_path = self._dir / "catalog.json"
        if cat_path.exists():
            self._catalog = Catalog.from_json(cat_path.read_text())
        else:
            self._catalog = Catalog()
        
        self._user_store = UserStore(self._dir / "users.json")
        self._user_store.ensure_default_admin()

        self._planner = Planner(self._catalog)
        self._executor = Executor(self._catalog, self._pool, self._txn, self._user_store)
        
    @property
    def user_store(self):
        """The authentication user store."""
        return self._user_store

    def execute(self, sql: str) -> dict:
        """Parse, plan, and execute a single SQL statement.
 
        This is the main entry point for all database operations.
        Errors at any stage (parse, plan, execute) are caught and
        returned as a message in the result dictionary rather than
        raised as exceptions.
 
        Parameters
        ----------
        sql : str
            A single SQL statement (with or without trailing ``;``).
 
        Returns
        -------
        dict
            ``{"columns": list[str], "rows": list[list], "message": str}``
 
            * **columns** — column names for SELECT results;
              empty for DDL/DML.
            * **rows** — result tuples for SELECT; empty for DDL/DML.
            * **message** — human-readable status (e.g. ``"3 row(s)"``
              or ``"Table 'users' created"``).
        """
        try:
            ast = parse_sql(sql)
        except ParseError as e:
            return {"columns": [], "rows": [], "message": f"Parse error: {e}"}
        
        try:
            plan = self._planner.plan(ast)
        except Exception as e:
            return {"columns": [], "rows": [], "message": f"Plan error: {e}"}
        
        try:
            return self._executor.execute(plan)
        except ExecutionError as e:
            return {"columns": [], "rows": [], "message": f"Execution error: {e}"}
        except Exception as e:
            return {"columns": [], "rows": [], "message": f"Error: {e}"}
        
    def _recover(self):
        """Replay the WAL to recover from a crash.

        Three passes:
        1. Analysis — scan all records to classify transactions.
        2. Redo — re-apply after-images for committed transactions
           whose pages are stale (record LSN > page LSN).
        3. Undo — restore before-images for uncommitted transactions.

        After recovery, flush all pages and truncate the WAL.
        """
        records = list(self._wal.iter_records())
        if not records:
            return

        # Analysis: classify transactions
        committed = set()
        aborted = set()
        updates: dict[int, list] = {}  # txn_id -> [records]
        for rec in records:
            if rec.rec_type == LogRecordType.COMMIT:
                committed.add(rec.txn_id)
            elif rec.rec_type == LogRecordType.ABORT:
                aborted.add(rec.txn_id)
            elif rec.rec_type == LogRecordType.UPDATE:
                updates.setdefault(rec.txn_id, []).append(rec)

        dirty = False

        # Redo: apply after-images for committed txns in LSN order
        for rec in records:
            if rec.rec_type == LogRecordType.UPDATE and rec.txn_id in committed:
                _, after = _parse_update_payload(rec.data)
                raw = self._disk.read_page(rec.page_id)
                page = SlottedPage.from_bytes(raw)
                if rec.lsn > page.lsn:
                    self._disk.write_page(rec.page_id, after)
                    dirty = True

        # Undo: restore before-images for uncommitted transactions
        # Process in reverse LSN order for correct undo sequencing
        uncommitted_updates = []
        for rec in records:
            if rec.rec_type == LogRecordType.UPDATE:
                if rec.txn_id not in committed and rec.txn_id not in aborted:
                    uncommitted_updates.append(rec)
        for rec in reversed(uncommitted_updates):
            before, _ = _parse_update_payload(rec.data)
            self._disk.write_page(rec.page_id, before)
            dirty = True

        if dirty:
            self._disk.flush()
        self._wal.truncate()

    def close(self):
        """Flush all state to disk and close file handles.

        Steps:

        1. Flush every dirty page from the buffer pool to ``data.db``.
        2. Serialise the catalog to ``catalog.json``.
        3. Truncate the WAL (all pages are durable).
        4. Close the WAL and data file handles.
        """
        self._pool.flush_all()
        cat_path = self._dir / "catalog.json"
        cat_path.write_text(self._catalog.to_json())
        self._wal.truncate()
        self._wal.close()
        self._disk.close()
