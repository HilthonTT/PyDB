<p align="center">
  <img src="assets/logo.svg" alt="PyDB" width="540">
</p>

<p align="center">
  A relational database engine built from scratch in Python — storage, indexes, transactions, SQL and networking, with zero dependencies.
</p>

---

## Features

| Layer        | What you get                                                                     |
| ------------ | -------------------------------------------------------------------------------- |
| Storage      | Slotted-page heap files with overflow chains, free-list page allocation          |
| Indexes      | Disk-resident B+Trees with split / merge / redistribute                          |
| Caching      | LRU-K(2) buffer pool, WAL-aware eviction                                         |
| Transactions | STEAL/NO-FORCE WAL, strict 2PL, crash recovery (redo/undo), undo on abort        |
| SQL          | SELECT, INSERT, UPDATE, DELETE, JOIN, GROUP BY/HAVING, ORDER BY, LIMIT, EXPLAIN  |
| Planning     | Cost-based SeqScan vs IndexScan using `ANALYZE` statistics                       |
| Execution    | Volcano-style pull iterators, k-way external merge sort                          |
| Network      | Length-prefixed TCP protocol, PBKDF2 auth, client library with connection pool   |

## Quick Start

Requires **Python 3.10+** (stdlib only).

```bash
python -m pydb.main --data mydb                        # interactive REPL
python -m pydb.main --data mydb --server --port 5433   # TCP server
```

```
pydb> CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
pydb> INSERT INTO users (name, age) VALUES ('Alice', 30), ('Bob', 25);
pydb> SELECT * FROM users WHERE age > 26;
+----+-------+-----+
| id | name  | age |
+----+-------+-----+
| 1  | Alice | 30  |
+----+-------+-----+
1 row(s)
```

REPL commands: `.help` `.tables` `.schema TABLE` `.indexes` `.server [PORT]` `.quit`

> The server creates a default `admin` / `admin` user on first start — change it with
> `ALTER USER admin SET PASSWORD '...'`.

## Client Library

```python
from pydb.client import Client, ConnectionPool

with Client("127.0.0.1", 5433, username="admin", password="admin") as db:
    db.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)")
    db.execute("INSERT INTO products (name, price) VALUES ('Widget', 9.99)")
    print(db.execute("SELECT * FROM products")["rows"])   # [[1, 'Widget', 9.99]]

# Thread-safe pool for web servers
pool = ConnectionPool("127.0.0.1", 5433, min_size=2, max_size=10,
                      username="admin", password="admin")
with pool.connection() as db:
    rows = db.execute("SELECT * FROM users")["rows"]
pool.close()
```

A complete stdlib-only REST API (CRUD on `/todos`, pooled connections, transactions) lives in
[`examples/todo_api.py`](examples/todo_api.py) — start the server, then `python examples/todo_api.py`.

## SQL Reference

```sql
-- DDL                       types: INTEGER|INT, FLOAT|REAL|DOUBLE, TEXT|VARCHAR|STRING, BOOLEAN|BOOL
CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL, ...);
CREATE [UNIQUE] INDEX idx ON t (col1, col2);
DROP TABLE t;

-- DML
INSERT INTO t (a, b) VALUES (1, 'x'), (2, 'y');
UPDATE t SET a = a + 1 WHERE b = 'x';
DELETE FROM t WHERE a > 10;

-- Queries
SELECT u.name, COUNT(*), AVG(o.total)
  FROM users u JOIN orders o ON u.id = o.user_id
  WHERE u.age BETWEEN 18 AND 65 AND u.name LIKE 'A%' AND o.status IN ('paid', 'shipped')
  GROUP BY u.name HAVING COUNT(*) > 5
  ORDER BY u.name ASC LIMIT 10 OFFSET 5;

-- Transactions, planning, users
BEGIN; ... COMMIT;  -- or ROLLBACK
ANALYZE users;                              -- collect stats for the cost-based planner
EXPLAIN SELECT * FROM users WHERE id = 1;   -- large table: IndexScan(users using pk_users) [est_rows=1, cost=4]
CREATE USER bob IDENTIFIED BY 'pw';  ALTER USER bob SET PASSWORD 'new';  DROP USER bob;
```

## Architecture

```
REPL (repl.py) / TCP Server (server.py) / Client (client.py)
Parser      parser.py    SQL text → AST
Planner     planner.py   AST → physical plan (cost-based index selection)
Executor    executor.py  Volcano pull iterators, DML, external sort (sorter.py)
Catalog     catalog.py   schemas, record/key encoding, statistics → catalog.json
Txn Mgr     txn.py       WAL (wal.py) + strict 2PL + undo on abort
B+Tree      btree.py     disk-resident indexes
Buffer Pool cache.py     LRU-K(2), flushes WAL before evicting dirty pages
Disk        storage.py   page I/O, free list
Pages       page.py      4 KiB slotted pages
```

**Design notes**

- **WAL** — every page mutation logs full-page before/after images; recovery redoes committed and undoes uncommitted transactions; the log is truncated on clean shutdown.
- **Encoding** — records are a null bitmap + fixed-width fields + length-prefixed TEXT; index keys are big-endian with sign-bit flips so byte order equals value order.
- **Locking** — page-level strict 2PL, locks held to commit/abort, 5 s timeout to break deadlocks.
- **Planning** — SeqScan cost = `page_count`; IndexScan cost = `tree_height + selectivity × row_count`. Without `ANALYZE`, 10 % selectivity per predicate is assumed.
- **Auth** — PBKDF2-HMAC-SHA256 (100k iterations, 16-byte salt) stored in `users.json`; TCP requires auth, the REPL bypasses it.

## Wire Protocol

Every message is a little-endian `u32` length followed by a UTF-8 payload (max 16 MiB).

1. Client sends `{"auth": {"username": "...", "password": "..."}}` → server replies `{"ok": true|false, "message": ...}` and closes on failure.
2. Client sends raw SQL → server replies `{"columns": [...], "rows": [[...]], "message": "2 row(s)"}`.
3. Send `.quit` to close gracefully.

## License

See [LICENSE](LICENSE).
