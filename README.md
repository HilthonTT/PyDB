# PyDB

A relational database engine built from scratch in Python. Implements the full storage-to-query stack without external database dependencies.

## Architecture

```
Interactive REPL (repl.py) / TCP Server (server.py)
    |
    v
SQL Parser (parser.py) -------- SQL text -> AST nodes
    |
    v
Query Planner (planner.py) ---- AST -> physical plan (with index selection)
    |
    v
Executor (executor.py) -------- Volcano-style pull iterator
    |
    v
Catalog (catalog.py) ---------- table/index definitions, record encoding
    |
    v
Transaction Mgr (txn.py) ------ ACID via WAL + strict 2PL locking
    |
    v
B+Tree (btree.py) ------------- disk-resident indexes with split/merge
    |
    v
Buffer Pool (cache.py) -------- LRU-K(2) page cache with pin counting
    |
    v
Disk Manager (storage.py) ----- page-granularity I/O, free-list allocation
    |
    v
Slotted Pages (page.py) ------- variable-length records in 4096-byte pages
```

## Features

- **Storage**: slotted-page heap files with overflow chains
- **Indexes**: B+Tree with split, merge, and redistribute balancing
- **Caching**: LRU-K(2) buffer pool with dirty-page tracking
- **Transactions**: STEAL/NO-FORCE WAL with full-page before/after images, strict two-phase locking, crash recovery (redo/undo) on startup
- **SQL**: recursive-descent parser supporting SELECT, INSERT, UPDATE, DELETE, JOINs, GROUP BY, ORDER BY, LIMIT, ANALYZE
- **Query planning**: cost-based index selection using table statistics (row counts, distinct-value counts, selectivity estimates)
- **Authentication**: username/password auth with PBKDF2-HMAC-SHA256, user management via SQL
- **Networking**: length-prefixed TCP wire protocol with auth handshake
- **Sorting**: k-way external merge sort for large result sets
- **REPL**: interactive shell with readline support and ASCII table output

## Getting Started

### Prerequisites

- Python 3.10+ (stdlib only, no third-party dependencies)

### Interactive REPL

```bash
python -m pydb.main --data mydb
```

This opens an interactive shell:

```
pydb> CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
Table 'users' created

pydb> INSERT INTO users (name, age) VALUES ('Alice', 30);
1 row(s) inserted

pydb> INSERT INTO users (name, age) VALUES ('Bob', 25);
1 row(s) inserted

pydb> SELECT * FROM users;
+----+-------+-----+
| id | name  | age |
+----+-------+-----+
| 1  | Alice | 30  |
| 2  | Bob   | 25  |
+----+-------+-----+
2 row(s)

pydb> .quit
```

REPL dot-commands:

| Command          | Description                    |
| ---------------- | ------------------------------ |
| `.help`          | Show available commands        |
| `.tables`        | List all tables                |
| `.schema TABLE`  | Show table schema and indexes  |
| `.indexes`       | List all indexes               |
| `.server [PORT]` | Start TCP server in background |
| `.quit`          | Exit                           |

### TCP Server

Start a standalone server:

```bash
python -m pydb.main --data mydb --server --port 5433
```

On first startup, a default user `admin` with password `admin` is created. Change this immediately:

```sql
ALTER USER admin SET PASSWORD 'your_secure_password';
```

### Client Library

Connect from application code (credentials required for TCP connections):

```python
from pydb.client import Client

with Client("127.0.0.1", 5433, username="admin", password="admin") as db:
    db.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)")
    db.execute("INSERT INTO products (name, price) VALUES ('Widget', 9.99)")

    result = db.execute("SELECT * FROM products")
    print(result["columns"])  # ['id', 'name', 'price']
    print(result["rows"])     # [[1, 'Widget', 9.99]]
```

### Connection Pool

For web servers and concurrent applications:

```python
from pydb.client import ConnectionPool

pool = ConnectionPool("127.0.0.1", 5433, min_size=2, max_size=10,
                      username="admin", password="admin")

# Thread-safe: each `connection()` call gets its own client
with pool.connection() as db:
    result = db.execute("SELECT * FROM users WHERE age > 25")
    for row in result["rows"]:
        print(row)

pool.close()
```

### Flask Example

```python
from flask import Flask, request, jsonify
from pydb.client import ConnectionPool

app = Flask(__name__)
pool = ConnectionPool("127.0.0.1", 5433, min_size=2, max_size=10,
                      username="admin", password="admin")

@app.route("/query", methods=["POST"])
def query():
    sql = request.json["sql"]
    with pool.connection() as db:
        result = db.execute(sql)
    return jsonify(result)
```

## SQL Support

### Data Definition

```sql
CREATE TABLE table_name (
    column_name TYPE [PRIMARY KEY] [NOT NULL],
    ...
);

DROP TABLE table_name;

CREATE INDEX index_name ON table_name (col1, col2);
CREATE UNIQUE INDEX index_name ON table_name (col1);
```

Supported types: `INTEGER`/`INT`, `FLOAT`/`REAL`/`DOUBLE`, `TEXT`/`VARCHAR`/`STRING`, `BOOLEAN`/`BOOL`

### Data Manipulation

```sql
INSERT INTO table_name (col1, col2) VALUES (val1, val2), (val3, val4);

UPDATE table_name SET col1 = expr WHERE condition;

DELETE FROM table_name WHERE condition;
```

### Queries

```sql
SELECT col1, col2 FROM table_name
  WHERE condition
  ORDER BY col1 ASC, col2 DESC
  LIMIT 10 OFFSET 5;

-- Joins
SELECT u.name, o.total
  FROM users u
  JOIN orders o ON u.id = o.user_id;

-- Aggregation
SELECT department, COUNT(*), AVG(salary)
  FROM employees
  GROUP BY department
  HAVING COUNT(*) > 5;

-- Expressions
SELECT * FROM products WHERE price BETWEEN 10 AND 50;
SELECT * FROM users WHERE name LIKE 'A%';
SELECT * FROM users WHERE age IN (25, 30, 35);
SELECT * FROM users WHERE email IS NOT NULL;
```

### User Management

```sql
CREATE USER username IDENTIFIED BY 'password';

ALTER USER username SET PASSWORD 'new_password';

DROP USER username;
```

### Transaction Control

```sql
BEGIN;
-- ... multiple statements ...
COMMIT;
-- or
ROLLBACK;
```

### Table Statistics

```sql
-- Compute accurate statistics for cost-based planning
ANALYZE users;
-- ANALYZE users: row_count=1000, page_count=12, id.ndv=1000, name.ndv=950

-- View query plan with cost estimates (requires ANALYZE first)
EXPLAIN SELECT * FROM users WHERE id = 1;
-- IndexScan(users using pk_users) [est_rows=1, cost=4]
```

After running `ANALYZE`, the planner uses per-column distinct-value counts to estimate selectivity and choose between sequential scans and index scans based on I/O cost.

### Other

```sql
EXPLAIN SELECT * FROM users WHERE id = 1;
```

## Wire Protocol

The TCP protocol uses length-prefixed messages (little-endian `u32` header):

### Authentication Handshake

The first message after connecting must be an auth request:

```
Client -> Server:
  [len][JSON: {"auth": {"username": "admin", "password": "secret"}}]

Server -> Client:
  [len][JSON: {"ok": true, "message": "Authenticated as 'admin'"}]
  -- or on failure --
  [len][JSON: {"ok": false, "message": "Authentication failed"}]
```

On failure the server closes the connection. On success, the SQL loop begins.

### SQL Messages

```
Client -> Server:
  [payload_len: u32, 4 bytes][SQL string: UTF-8, payload_len bytes]

Server -> Client:
  [payload_len: u32, 4 bytes][JSON result: UTF-8, payload_len bytes]
```

JSON result format:

```json
{
  "columns": ["id", "name", "age"],
  "rows": [
    [1, "Alice", 30],
    [2, "Bob", 25]
  ],
  "message": "2 row(s)"
}
```

Send `.quit` to close the connection gracefully. Maximum message size: 16 MiB.

## Project Structure

```
pydb/
  __init__.py    -- global constants (PAGE_SIZE, BTREE_ORDER, etc.)
  page.py        -- slotted-page storage format, RID, overflow pages
  storage.py     -- disk manager (page I/O, free-list allocation)
  cache.py       -- LRU-K(2) buffer pool with pin counting
  btree.py       -- B+Tree indexes (insert, delete, range scan)
  wal.py         -- write-ahead log (append, recover, iterate, truncate)
  txn.py         -- transaction manager (strict 2PL, WAL logging, undo on abort)
  catalog.py     -- table/index definitions, record encoding, table statistics
  auth.py        -- user authentication (PBKDF2 hashing, user store)
  parser.py      -- recursive-descent SQL parser
  planner.py     -- cost-based query planner with index selection
  executor.py    -- Volcano-style pull executor
  sorter.py      -- k-way external merge sort
  engine.py      -- Database facade (wires all components together)
  server.py      -- TCP server and wire protocol
  client.py      -- client library with connection pooling
  repl.py        -- interactive REPL with readline support
  main.py        -- CLI entry point
```

## Design Decisions

- **STEAL/NO-FORCE WAL**: dirty pages can be flushed before commit; pages are not forced at commit time. Every page mutation (heap and B+Tree) logs full-page before/after images to the WAL. On crash, recovery redoes committed transactions and undoes uncommitted ones. The WAL is truncated on clean shutdown.
- **WAL-aware buffer pool**: the buffer pool flushes the WAL before evicting dirty pages, ensuring the write-ahead invariant holds even under memory pressure.
- **Undo on abort**: when a transaction rolls back, all its page modifications are reversed by replaying WAL UPDATE records in reverse order, restoring before-images.
- **Slotted pages**: slot directory grows forward, record payloads grow backward. Deletion tombstones slots; `compact()` reclaims space.
- **Record encoding**: null bitmap + fixed-width fields (INTEGER=8B, FLOAT=8B, BOOLEAN=1B) + length-prefixed TEXT.
- **Key encoding**: big-endian with sign-bit flips for correct lexicographic sort order.
- **Strict 2PL**: all locks held until commit/abort. Timeout-based deadlock prevention (5s).
- **Cost-based planning**: the planner estimates I/O cost for sequential scans (`page_count`) vs index scans (`tree_height + selectivity * row_count`). Selectivity is derived from per-column distinct-value counts computed by `ANALYZE`. Without statistics, a default selectivity of 10% per predicate is assumed.
- **Catalog**: persisted as `catalog.json` (including table statistics), not in the data file.
- **Authentication**: passwords hashed with PBKDF2-HMAC-SHA256 (100k iterations, 16-byte salt). Users stored in `users.json`. TCP connections require auth; REPL sessions bypass it.
