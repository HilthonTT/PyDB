# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyDB is a relational database engine built from scratch in Python. It implements the full storage-to-query stack without external database dependencies.

## Running

```bash
python -m pydb.main --data pydb_data --port 5433
```

No build step or package manager config exists yet. The project is pure Python with no third-party dependencies (only stdlib: `struct`, `threading`, `json`, `zlib`, `re`, `enum`, `dataclasses`).

## Architecture

The system is layered bottom-up. Each layer only depends on layers below it:

```
Parser (parser.py) ─── SQL text → AST nodes
    ↕
Catalog (catalog.py) ─── table/index definitions, record encoding/decoding
    ↕
Transaction Manager (txn.py) ─── ACID via WAL + strict 2PL locking
    ↕
B+Tree (btree.py) ─── disk-resident indexes with split/merge/redistribute
    ↕
Buffer Pool (cache.py) ─── LRU-K(2) page cache with pin counting
    ↕
Disk Manager (storage.py) ─── page-granularity I/O, free-list allocation
    ↕
Slotted Pages (page.py) ─── variable-length records in fixed 4096-byte pages
```

**Global constants** live in `pydb/__init__.py` (PAGE_SIZE=4096, BTREE_ORDER=128, BUFFER_POOL_CAP=1024, etc.).

### Key design decisions

- **STEAL/NO-FORCE WAL** — dirty pages can be flushed before commit (steal), and pages are not forced to disk at commit time (no-force). Recovery replays the WAL.
- **Slotted pages** — slot directory grows forward, record payloads grow backward. Deletion tombstones slots; `compact()` reclaims space.
- **B+Tree nodes** are serialized into slot 0 of their SlottedPage. Internal nodes store `[child0]{key, child}*`; leaves store `{key, RID}*` with a `next_leaf` chain pointer.
- **Record encoding** uses a null bitmap followed by fixed-width fields (INTEGER=8B, FLOAT=8B, BOOLEAN=1B) and length-prefixed TEXT. Key encoding uses big-endian with sign-bit flips for correct lexicographic sort order.
- **Catalog** is persisted as `catalog.json` in the database directory, not in the data file itself.
- **Strict 2PL** with timeout-based deadlock prevention (5s default). Lock granularity is page-level.

### Components not yet implemented

The `__init__.py` docstring describes planned components that don't exist in code yet: cost-based query planner, Volcano-style executor, k-way external merge sort, TCP wire protocol, and interactive REPL. The parser currently only has the tokenizer and token types — the recursive-descent parser and AST node definitions are not yet implemented.
