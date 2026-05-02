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
 
import os
from pathlib import Path
 
from pydb import BUFFER_POOL_CAP
from pydb.storage import DiskManager
from pydb.cache import BufferPool
from pydb.wal import WAL
from pydb.txn import TransactionManager
from pydb.catalog import Catalog
from pydb.parser import parse_sql, ParseError
from pydb.planner import Planner
from pydb.executor import Executor, ExecutionError

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
        self._pool = BufferPool(self._disk, capacity=buffer_pool_size)
        self._wal = WAL(self._dir / "wal.log")
        self._txn = TransactionManager(self._wal, self._pool)
        
        cat_path = self._dir / "catalog.json"
        if cat_path.exists():
            self._catalog = Catalog.from_json(cat_path.read_text())
        else:
            self._catalog = Catalog()
        
        self._planner = Planner(self._catalog)
        self._executor = Executor(self._catalog, self._pool, self._txn)
        
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
        
    def close(self):
        """Flush all state to disk and close file handles.

        Steps:

        1. Flush every dirty page from the buffer pool to ``data.db``.
        2. Serialise the catalog to ``catalog.json``.
        3. Close the WAL file handle.
        4. Close the data file handle.
        """
        self._pool.flush_all()
        cat_path = self._dir / "catalog.json"
        cat_path.write_text(self._catalog.to_json())
        self._wal.close()
        self._disk.close()
