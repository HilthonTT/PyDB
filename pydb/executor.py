"""
Query Executor
==============
 
Overview
--------
The executor is the runtime engine that takes a physical query plan
(from the planner) and produces actual results by reading and writing
data through the buffer pool and B+Tree indexes.
 
Execution model — Volcano / Iterator / Pull
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each plan node is compiled into a **Python generator** that yields
rows one at a time.  The top-level consumer (``execute``) calls
``list(self._pull(plan, txn))`` which pulls rows through the
pipeline:
 
* ``SeqScan``   — iterates heap pages, decodes records, applies filter.
* ``IndexScan`` — walks B+Tree leaf pages, fetches matching records.
* ``NestedLoopJoin`` — for each left row, scans all right rows.
* ``Filter``    — yields only rows passing the predicate.
* ``Projection``— evaluates column expressions per row.
* ``SortNode``  — materialises all input, sorts (in-memory or external).
* ``LimitNode`` — stops after *N* rows (with optional offset).
* ``AggregateNode`` — groups rows and computes aggregate functions.

This model is memory-efficient: most operators stream without
materialising the full result set.  Only ``SortNode`` and
``AggregateNode`` (with ``GROUP BY``) need to buffer all input.
 
Expression evaluation
~~~~~~~~~~~~~~~~~~~~~
The executor includes a recursive expression evaluator
(``_eval_expr``) that handles all AST expression types:
``ColumnRef``, ``Literal``, ``BinaryOp``, ``UnaryOp``,
``IsNullExpr``, ``InExpr``, ``BetweenExpr``, and ``FuncCall``.
 
Column references are resolved via a **context dictionary** built
from the current row and the table definition.  Keys include both
bare column names (``"age"``) and qualified names (``"users.age"``),
plus any alias (``"u.age"``).
 
Heap management
~~~~~~~~~~~~~~~
Tables store their rows in a chain of ``SlottedPage`` objects linked
via ``overflow_pid``.  ``_scan_heap`` walks this chain, decoding
every live record.  ``_heap_insert`` finds a page with free space
(or allocates a new one and chains it) and inserts the encoded
record.
 
Index maintenance
~~~~~~~~~~~~~~~~~
On ``INSERT`` and ``DELETE``, the executor updates **every** index
defined on the table.  For each index, it encodes the indexed column
values into a B+Tree key (via ``catalog.encode_key``) and calls
``tree.insert`` or ``tree.delete``.
 
Transaction integration
~~~~~~~~~~~~~~~~~~~~~~~
Every statement is wrapped in a transaction.  If no explicit
``BEGIN`` is active, the executor creates an **auto-commit**
transaction that commits on success or aborts on error.  Explicit
transactions span multiple statements and are committed/aborted
by ``COMMIT``/``ROLLBACK``.
"""

from __future__ import annotations

import fnmatch
import threading
from typing import Any, Iterator, Optional

from pydb import INVALID_PAGE
from pydb.page import PageType, RID
from pydb.cache import BufferPool
from pydb.btree import BPlusTree
from pydb.catalog import (
    Catalog, TableDef, TableStats, Column, ColType,
    encode_record, decode_record, encode_key, parse_col_type,
)
from pydb.txn import TransactionManager, Transaction, LockMode
from pydb.planner import (
    SeqScan, IndexScan, NestedLoopJoin, Filter, Projection,
    SortNode, LimitNode, AggregateNode,
    InsertPlan, UpdatePlan, DeletePlan,
    CreateTablePlan, DropTablePlan, CreateIndexPlan,
    BeginPlan, CommitPlan, RollbackPlan, ExplainPlan,
    CreateUserPlan, DropUserPlan, AlterUserPlan,
    AnalyzePlan,
    format_plan,
)
from pydb.parser import (
    ColumnRef, Literal, BinaryOp, UnaryOp, FuncCall,
    IsNullExpr, InExpr, BetweenExpr,
)
from pydb.sorter import ExternalMergeSorter

class _NullSentinel:
    """Sentinel for NULL values in sort keys. Sorts before/after all real values."""
    __slots__ = ("_first",)
    def __init__(self, first: bool = True):
        self._first = first
    def __lt__(self, other):
        if isinstance(other, _NullSentinel):
            return False
        return self._first
    def __gt__(self, other):
        if isinstance(other, _NullSentinel):
            return False
        return not self._first
    def __le__(self, other):
        return not self.__gt__(other)
    def __ge__(self, other):
        return not self.__lt__(other)
    def __eq__(self, other):
        return isinstance(other, _NullSentinel)

class _ReverseKey:
    """Wrapper that reverses comparison order for DESC sort columns."""
    __slots__ = ("val",)
    def __init__(self, val):
        self.val = val
    def __lt__(self, other):
        if isinstance(other, _NullSentinel):
            return not other.__lt__(self)
        if isinstance(other, _ReverseKey):
            return self.val > other.val
        return self.val > other
    def __gt__(self, other):
        if isinstance(other, _NullSentinel):
            return not other.__gt__(self)
        if isinstance(other, _ReverseKey):
            return self.val < other.val
        return self.val < other
    def __le__(self, other):
        return not self.__gt__(other)
    def __ge__(self, other):
        return not self.__lt__(other)
    def __eq__(self, other):
        if isinstance(other, _ReverseKey):
            return self.val == other.val
        return False

class ExecutionError(Exception):
    pass

class Executor:
    """Executes physical query plans against the storage engine.
 
    The executor is stateful — it tracks the currently active
    explicit transaction (if any) so that ``BEGIN`` / ``COMMIT`` /
    ``ROLLBACK`` work correctly across multiple ``execute`` calls.
 
    Parameters
    ----------
    catalog : Catalog
        The system catalog (table/index metadata).
    pool : BufferPool
        The buffer pool for all page I/O.
    txn_mgr : TransactionManager
        The transaction manager for locking and WAL logging.
    """
    
    def __init__(self, catalog: Catalog, pool: BufferPool, txn_mgr: TransactionManager,
                 user_store=None):
        self._cat = catalog
        self._pool = pool
        self._txn = txn_mgr
        self._user_store = user_store
        self._local = threading.local()
        
    def execute(self, plan) -> dict:
        """Execute a plan and return the result.
 
        Handles transaction control plans (``BEGIN``, ``COMMIT``,
        ``ROLLBACK``) directly.  All other plans are wrapped in an
        auto-commit transaction if no explicit transaction is active.
 
        Parameters
        ----------
        plan : plan node
            A physical plan node from the ``Planner``.
 
        Returns
        -------
        dict
            ``{"columns": list[str], "rows": list[list], "message": str}``
        """
        # transaction control
        if isinstance(plan, BeginPlan):
            txn = self._txn.begin()
            self._local.current_txn = txn
            return {"columns": [], "rows": [], "message": f"BEGIN (txn {txn.txn_id})"}

        if isinstance(plan, CommitPlan):
            cur = getattr(self._local, 'current_txn', None)
            if cur:
                self._txn.commit(cur)
                self._local.current_txn = None
            return {"columns": [], "rows": [], "message": "COMMIT"}

        if isinstance(plan, RollbackPlan):
            cur = getattr(self._local, 'current_txn', None)
            if cur:
                self._txn.abort(cur)
                self._local.current_txn = None
            return {"columns": [], "rows": [], "message": "ROLLBACK"}

        if isinstance(plan, CreateUserPlan):
            self._user_store.create_user(plan.username, plan.password)
            return {"columns": [], "rows": [], "message": f"User '{plan.username}' created"}

        if isinstance(plan, DropUserPlan):
            self._user_store.drop_user(plan.username)
            return {"columns": [], "rows": [], "message": f"User '{plan.username}' dropped"}

        if isinstance(plan, AlterUserPlan):
            self._user_store.alter_password(plan.username, plan.new_password)
            return {"columns": [], "rows": [], "message": f"Password updated for '{plan.username}'"}

        if isinstance(plan, AnalyzePlan):
            return self._exec_analyze(plan)

        # wrap in auto-transaction if no explicit txn
        cur = getattr(self._local, 'current_txn', None)
        auto = cur is None
        txn = cur or self._txn.begin()
        try:
            result = self._exec(plan, txn)
            if auto:
                self._txn.commit(txn)
            return result
        except Exception:
            if auto:
                self._txn.abort(txn)
            raise
        
    def _exec(self, plan, txn: Transaction) -> dict:
        if isinstance(plan, ExplainPlan):
            text = format_plan(plan.child, catalog=self._cat)
            return {"columns": ["plan"], "rows": [[text]], "message": "EXPLAIN"}
        
        if isinstance(plan, CreateTablePlan):
            return self._exec_create_table(plan, txn)
        
        if isinstance(plan, DropTablePlan):
            return self._exec_drop_table(plan, txn)
        
        if isinstance(plan, CreateIndexPlan):
            return self._exec_create_index(plan, txn)
        
        if isinstance(plan, InsertPlan):
            return self._exec_insert(plan, txn)
        
        if isinstance(plan, UpdatePlan):
            return self._exec_update(plan, txn)
        
        if isinstance(plan, DeletePlan):
            return self._exec_delete(plan, txn)
        
        # SELECT (query plan tree) — pull rows
        rows = list(self._pull(plan, txn))
        cols = self._infer_columns(plan)
        return {"columns": cols, "rows": rows, "message": f"{len(rows)} row(s)"}
    
    def _exec_create_table(self, plan: CreateTablePlan, txn: Transaction) -> dict:
        stmt = plan.stmt
        columns: list[Column] = []
        
        for cname, ctype_str, nullable, pk in stmt.columns:
            ct = parse_col_type(ctype_str)
            columns.append(Column(cname, ct, nullable, pk))
            
        # allocate first heap page
        first_page = self._pool.new_page()
        before = first_page.to_bytes()
        first_page.page_type = PageType.DATA
        after = first_page.to_bytes()
        pid = first_page.page_id
        lsn = self._txn.log_update(txn, pid, before, after)
        first_page.lsn = lsn
        first_page._write_header()
        self._pool.unpin(pid, dirty=True)
        
        tdef = TableDef(stmt.table, columns, heap_page=pid)
        self._cat.create_table(tdef)
        
        # auto-create PK index
        pk_cols = [c.name for c in columns if c.primary_key]
        if pk_cols:
            idx_name = f"pk_{stmt.table}"
            tree = BPlusTree(self._pool, txn=txn, txn_mgr=self._txn)
            root = tree.create()
            from pydb.catalog import IndexDef
            idef = IndexDef(idx_name, stmt.table, pk_cols, root, unique=True)
            self._cat.create_index(idef)
            
        return {"columns": [], "rows": [], "message": f"Table '{stmt.table}' created"}

    def _exec_drop_table(self, plan: DropTablePlan, txn: Transaction) -> dict:
        table_name = plan.table
        tdef = self._cat.get_table(table_name)
        self._txn.acquire(txn, tdef.heap_page, LockMode.EXCLUSIVE)

        # Deallocate all heap pages
        pid = tdef.heap_page
        while pid != INVALID_PAGE:
            page = self._pool.fetch_page(pid)
            next_pid = page.overflow_pid
            self._pool.unpin(pid)
            self._pool.delete_page(pid)
            pid = next_pid

        # Deallocate index root pages
        for idx in tdef.indexes.values():
            if idx.root_page != INVALID_PAGE:
                self._pool.delete_page(idx.root_page)

        self._cat.drop_table(table_name)
        return {"columns": [], "rows": [], "message": f"Table '{table_name}' dropped"}

    def _exec_create_index(self, plan: CreateIndexPlan, txn: Transaction) -> dict:
        stmt = plan.stmt
        tdef = self._cat.get_table(stmt.table)
        tree = BPlusTree(self._pool, txn=txn, txn_mgr=self._txn)
        root = tree.create()
        from pydb.catalog import IndexDef
        idef = IndexDef(stmt.index_name, stmt.table, stmt.columns, root, stmt.unique)
        self._cat.create_index(idef)
        
        # back-fill existing rows
        idx_col_defs = [tdef.columns[tdef.col_index(c)] for c in stmt.columns]
        for rid, row in self._scan_heap(tdef, txn):
            key_vals = [row[tdef.col_index(c)] for c in stmt.columns]
            key = encode_key(idx_col_defs, key_vals)
            tree.insert(key, rid)
 
        return {"columns": [], "rows": [],
                "message": f"Index '{stmt.index_name}' created on {stmt.table}({', '.join(stmt.columns)})"}
        
    def _exec_insert(self, plan: InsertPlan, txn: Transaction) -> dict:
        tdef = self._cat.get_table(plan.table)
        self._txn.acquire(txn, tdef.heap_page, LockMode.EXCLUSIVE)
 
        cols = plan.columns if plan.columns else tdef.col_names
        count = 0
        for row_exprs in plan.rows:
            # evaluate expressions
            raw_vals = [self._eval_expr(e, {}, []) for e in row_exprs]
            # map to full column order
            values = [None] * len(tdef.columns)
            for i, cname in enumerate(cols):
                ci = tdef.col_index(cname)
                values[ci] = raw_vals[i]
 
            # auto-generate PK if integer pk with no value
            for i, c in enumerate(tdef.columns):
                if c.primary_key and c.col_type == ColType.INTEGER and values[i] is None:
                    values[i] = tdef.next_rowid
                    tdef.next_rowid += 1
 
            record = encode_record(tdef.columns, values)
            rid = self._heap_insert(tdef, record, txn)
 
            # update indexes
            for idx in tdef.indexes.values():
                idx_cols = [tdef.columns[tdef.col_index(c)] for c in idx.columns]
                key_vals = [values[tdef.col_index(c)] for c in idx.columns]
                key = encode_key(idx_cols, key_vals)
                tree = BPlusTree(self._pool, idx.root_page, txn, self._txn)
                tree.insert(key, rid)
                idx.root_page = tree.root_pid
 
            count += 1
 
        tdef.stats.row_count += count
        return {"columns": [], "rows": [], "message": f"{count} row(s) inserted"}

    def _exec_update(self, plan: UpdatePlan, txn: Transaction) -> dict:
        tdef = self._cat.get_table(plan.table)
        self._txn.acquire(txn, tdef.heap_page, LockMode.EXCLUSIVE)
        count = 0

        # Collect (RID, old_row) pairs matching the filter
        rows_to_update = []
        for rid, row in self._scan_heap(tdef, txn):
            if plan.scan.filter:
                ctx = self._row_context(tdef, row)
                if not self._eval_expr(plan.scan.filter, ctx, []):
                    continue
            rows_to_update.append((rid, row))

        for rid, row in rows_to_update:
            new_vals = list(row)
            for col_name, expr in plan.assignments:
                ci = tdef.col_index(col_name)
                ctx = {c.name.lower(): row[i] for i, c in enumerate(tdef.columns)}
                new_vals[ci] = self._eval_expr(expr, ctx, [])
            new_record = encode_record(tdef.columns, new_vals)

            # Delete old record
            page = self._pool.fetch_page(rid.page_id)
            before = page.to_bytes()
            page.delete(rid.slot_idx)
            after = page.to_bytes()
            lsn = self._txn.log_update(txn, rid.page_id, before, after)
            page.lsn = lsn
            page._write_header()
            self._pool.unpin(rid.page_id, dirty=True)

            # Insert new record
            new_rid = self._heap_insert(tdef, new_record, txn)

            # Update indexes: remove old key, insert new key
            for idx in tdef.indexes.values():
                idx_cols = [tdef.columns[tdef.col_index(c)] for c in idx.columns]
                old_key_vals = [row[tdef.col_index(c)] for c in idx.columns]
                old_key = encode_key(idx_cols, old_key_vals)
                new_key_vals = [new_vals[tdef.col_index(c)] for c in idx.columns]
                new_key = encode_key(idx_cols, new_key_vals)
                tree = BPlusTree(self._pool, idx.root_page, txn, self._txn)
                tree.delete(old_key, rid)
                tree.insert(new_key, new_rid)
                idx.root_page = tree.root_pid
            count += 1
 
        return {"columns": [], "rows": [], "message": f"{count} row(s) updated"}
 
    def _exec_delete(self, plan: DeletePlan, txn: Transaction) -> dict:
        tdef = self._cat.get_table(plan.table)
        self._txn.acquire(txn, tdef.heap_page, LockMode.EXCLUSIVE)
        count = 0
 
        rids_to_delete = []
        for rid, row in self._scan_heap(tdef, txn):
            if plan.scan.filter:
                ctx = {c.name.lower(): row[i] for i, c in enumerate(tdef.columns)}
                if not self._eval_expr(plan.scan.filter, ctx, []):
                    continue
            rids_to_delete.append((rid, row))
 
        for rid, row in rids_to_delete:
            page = self._pool.fetch_page(rid.page_id)
            before = page.to_bytes()
            page.delete(rid.slot_idx)
            after = page.to_bytes()
            lsn = self._txn.log_update(txn, rid.page_id, before, after)
            page.lsn = lsn
            page._write_header()
            self._pool.unpin(rid.page_id, dirty=True)
            # remove from indexes
            for idx in tdef.indexes.values():
                idx_cols = [tdef.columns[tdef.col_index(c)] for c in idx.columns]
                key_vals = [row[tdef.col_index(c)] for c in idx.columns]
                key = encode_key(idx_cols, key_vals)
                tree = BPlusTree(self._pool, idx.root_page, txn, self._txn)
                tree.delete(key, rid)
                idx.root_page = tree.root_pid
            count += 1
 
        tdef.stats.row_count -= count
        return {"columns": [], "rows": [], "message": f"{count} row(s) deleted"}

    def _exec_analyze(self, plan: AnalyzePlan) -> dict:
        """Compute accurate statistics for a table by scanning the heap."""
        tdef = self._cat.get_table(plan.table)
        txn = self._txn.begin()
        try:
            row_count = 0
            page_count = 0
            distinct_sets: dict[str, set] = {c.name.lower(): set() for c in tdef.columns}

            pid = tdef.heap_page
            while pid != INVALID_PAGE:
                page_count += 1
                page = self._pool.fetch_page(pid)
                for _, rec_bytes in page.iter_records():
                    row = decode_record(tdef.columns, rec_bytes)
                    row_count += 1
                    for i, col in enumerate(tdef.columns):
                        if row[i] is not None:
                            distinct_sets[col.name.lower()].add(row[i])
                next_pid = page.overflow_pid
                self._pool.unpin(pid)
                pid = next_pid

            tdef.stats = TableStats(
                row_count=row_count,
                page_count=max(page_count, 1),
                distinct_counts={k: len(v) for k, v in distinct_sets.items()},
            )
            self._txn.commit(txn)
        except Exception:
            self._txn.abort(txn)
            raise

        parts = [f"row_count={tdef.stats.row_count}",
                 f"page_count={tdef.stats.page_count}"]
        for col, ndv in sorted(tdef.stats.distinct_counts.items()):
            parts.append(f"{col}.ndv={ndv}")
        return {"columns": [], "rows": [],
                "message": f"ANALYZE {plan.table}: {', '.join(parts)}"}

    def _pull(self, node, txn: Transaction) -> Iterator:
        if isinstance(node, SeqScan):
            yield from self._pull_seq_scan(node, txn)
        elif isinstance(node, IndexScan):
            yield from self._pull_index_scan(node, txn)
        elif isinstance(node, NestedLoopJoin):
            yield from self._pull_nlj(node, txn)
        elif isinstance(node, Filter):
            yield from self._pull_filter(node, txn)
        elif isinstance(node, Projection):
            yield from self._pull_projection(node, txn)
        elif isinstance(node, SortNode):
            yield from self._pull_sort(node, txn)
        elif isinstance(node, LimitNode):
            yield from self._pull_limit(node, txn)
        elif isinstance(node, AggregateNode):
            yield from self._pull_aggregate(node, txn)
            
    def _pull_seq_scan(self, node: SeqScan, txn: Transaction):
        tdef = self._cat.get_table(node.table)
        self._txn.acquire(txn, tdef.heap_page, LockMode.SHARED)
        for rid, row in self._scan_heap(tdef, txn):
            if node.filter:
                ctx = self._row_context(tdef, row, node.alias)
                if not self._eval_expr(node.filter, ctx, []):
                    continue
            yield row
 
    def _pull_index_scan(self, node: IndexScan, txn: Transaction):
        tdef = self._cat.get_table(node.table)
        idef = self._cat.get_index(node.index_name)
        tree = BPlusTree(self._pool, idef.root_page)
 
        if node.lookup_key and isinstance(node.lookup_key, dict):
            # point lookup
            idx_cols = [tdef.columns[tdef.col_index(c)] for c in idef.columns]
            vals = [node.lookup_key.get(c.lower()) for c in idef.columns
                    if c.lower() in node.lookup_key]
            key_prefix = encode_key(idx_cols[:len(vals)],
                                    vals)
            for k, rid in tree.range_scan(key_prefix, None):
                if not k.startswith(key_prefix):
                    break
                page = self._pool.fetch_page(rid.page_id)
                rec = page.read(rid.slot_idx)
                self._pool.unpin(rid.page_id)
                if rec is None:
                    continue
                row = decode_record(tdef.columns, rec)
                if node.filter:
                    ctx = self._row_context(tdef, row, node.alias)
                    if not self._eval_expr(node.filter, ctx, []):
                        continue
                yield row
        else:
            # full index scan
            for k, rid in tree.range_scan():
                page = self._pool.fetch_page(rid.page_id)
                rec = page.read(rid.slot_idx)
                self._pool.unpin(rid.page_id)
                if rec is None:
                    continue
                row = decode_record(tdef.columns, rec)
                if node.filter:
                    ctx = self._row_context(tdef, row, node.alias)
                    if not self._eval_expr(node.filter, ctx, []):
                        continue
                yield row
 
    def _pull_nlj(self, node: NestedLoopJoin, txn: Transaction):
        left_rows = list(self._pull(node.left, txn))
        l_tdef = self._scan_tdef(node.left)
        r_tdef = self._scan_tdef(node.right)
 
        for lrow in left_rows:
            matched = False
            for rrow in self._pull(node.right, txn):
                combined = list(lrow) + list(rrow)
                if node.on:
                    ctx = {}
                    if l_tdef:
                        ctx.update(self._row_context(l_tdef, lrow,
                                                      getattr(node.left, 'alias', None)))
                    if r_tdef:
                        ctx.update(self._row_context(r_tdef, rrow,
                                                      getattr(node.right, 'alias', None)))
                    if not self._eval_expr(node.on, ctx, []):
                        continue
                matched = True
                yield combined
            if not matched and node.join_type == "LEFT":
                yield list(lrow) + [None] * (len(r_tdef.columns) if r_tdef else 0)
 
    def _pull_filter(self, node: Filter, txn: Transaction):
        for row in self._pull(node.child, txn):
            if self._eval_expr(node.predicate, {}, row):
                yield row
 
    def _pull_projection(self, node: Projection, txn: Transaction):
        if node.child is None:
            # expression-only SELECT (no FROM)
            row = [self._eval_expr(c, {}, []) for c in node.columns]
            yield row
            return
 
        tdef = self._scan_tdef(node.child)
        # Check if child is an AggregateNode — rows are already aggregated
        is_agg_child = isinstance(node.child, AggregateNode)
        for row in self._pull(node.child, txn):
            if any(isinstance(c, Literal) and c.value == "*" for c in node.columns):
                yield row
            else:
                if is_agg_child:
                    # Row is [group_vals..., agg_vals...] — reorder to match SELECT list
                    agg_node = node.child
                    group_exprs = agg_node.group_by
                    agg_exprs = agg_node.aggregates
                    # Build name-based index for group-by columns
                    group_idx = {}
                    for gi, g in enumerate(group_exprs):
                        if isinstance(g, ColumnRef):
                            group_idx[g.name.lower()] = gi
                    # Build index for aggregate positions
                    agg_offset = len(group_exprs)
                    out = []
                    ai = 0
                    for c in node.columns:
                        if isinstance(c, FuncCall):
                            out.append(row[agg_offset + ai])
                            ai += 1
                        elif isinstance(c, ColumnRef):
                            # Match by name to the correct group-by position
                            pos = group_idx.get(c.name.lower())
                            if pos is not None:
                                out.append(row[pos])
                            else:
                                ctx = self._row_context(tdef, row) if tdef else {}
                                out.append(self._eval_expr(c, ctx, row))
                        else:
                            ctx = self._row_context(tdef, row) if tdef else {}
                            out.append(self._eval_expr(c, ctx, row))
                    yield out
                else:
                    out = []
                    for c in node.columns:
                        ctx = self._row_context(tdef, row) if tdef else {}
                        out.append(self._eval_expr(c, ctx, row))
                    yield out
 
    def _pull_sort(self, node: SortNode, txn: Transaction):
        tdef = self._scan_tdef(node.child)
        rows = list(self._pull(node.child, txn))

        order_specs = node.order_by

        def make_key(row):
            parts = []
            for expr, direction in order_specs:
                ctx = self._row_context(tdef, row) if tdef else {}
                val = self._eval_expr(expr, ctx, row)
                if val is None:
                    # NULLs sort first in ASC, last in DESC
                    part = _NullSentinel(first=(direction != "DESC"))
                elif direction == "DESC":
                    part = _ReverseKey(val)
                else:
                    part = val
                parts.append(part)
            return tuple(parts)

        if len(rows) <= 10_000:
            rows.sort(key=make_key)
            yield from rows
        else:
            sorter = ExternalMergeSorter(key_func=make_key)
            yield from sorter.sort(iter(rows))

    def _pull_limit(self, node: LimitNode, txn: Transaction):
        count = 0
        skipped = 0
        for row in self._pull(node.child, txn):
            if skipped < node.offset:
                skipped += 1
                continue
            yield row
            count += 1
            if count >= node.limit:
                return
 
    def _pull_aggregate(self, node: AggregateNode, txn: Transaction):
        tdef = self._scan_tdef(node.child)
        rows = list(self._pull(node.child, txn))
 
        if not node.group_by:
            # single-group aggregation
            result = self._compute_aggs(node.aggregates, rows, tdef)
            yield result
            return
 
        # group by
        groups: dict[tuple, list] = {}
        for row in rows:
            ctx = self._row_context(tdef, row) if tdef else {}
            gkey = tuple(self._eval_expr(g, ctx, row) for g in node.group_by)
            groups.setdefault(gkey, []).append(row)
 
        for gkey, grows in groups.items():
            agg_vals = self._compute_aggs(node.aggregates, grows, tdef)
            result = list(gkey) + agg_vals
            if node.having:
                ctx = self._row_context(tdef, grows[0]) if tdef else {}
                # Inject computed aggregate values so HAVING can reference them
                agg_ctx = {}
                for agg_expr, agg_val in zip(node.aggregates, agg_vals):
                    if isinstance(agg_expr, FuncCall):
                        agg_ctx[self._agg_key(agg_expr)] = agg_val
                if not self._eval_having(node.having, ctx, agg_ctx, result):
                    continue
            yield result
 
    def _compute_aggs(self, aggs: list, rows: list, tdef) -> list:
        result = []
        for agg in aggs:
            if not isinstance(agg, FuncCall):
                continue
            fname = agg.name.upper()
            if fname == "COUNT":
                if agg.args and isinstance(agg.args[0], Literal) and agg.args[0].value == "*":
                    result.append(len(rows))
                else:
                    col_vals = self._extract_col_values(agg.args[0], rows, tdef)
                    result.append(sum(1 for v in col_vals if v is not None))
            elif fname == "SUM":
                col_vals = self._extract_col_values(agg.args[0], rows, tdef)
                result.append(sum(v for v in col_vals if v is not None))
            elif fname == "AVG":
                col_vals = [v for v in self._extract_col_values(agg.args[0], rows, tdef)
                            if v is not None]
                result.append(sum(col_vals) / len(col_vals) if col_vals else None)
            elif fname == "MIN":
                col_vals = [v for v in self._extract_col_values(agg.args[0], rows, tdef)
                            if v is not None]
                result.append(min(col_vals) if col_vals else None)
            elif fname == "MAX":
                col_vals = [v for v in self._extract_col_values(agg.args[0], rows, tdef)
                            if v is not None]
                result.append(max(col_vals) if col_vals else None)
        return result
 
    def _agg_key(self, func: FuncCall) -> str:
        """Build a canonical string key for an aggregate expression."""
        args_str = ", ".join(
            ("*" if (isinstance(a, Literal) and a.value == "*") else
             (a.name if isinstance(a, ColumnRef) else "?"))
            for a in func.args
        )
        return f"{func.name.upper()}({args_str})"

    def _eval_having(self, expr, ctx: dict, agg_ctx: dict, row: list) -> Any:
        """Evaluate a HAVING predicate with aggregate values available."""
        if isinstance(expr, FuncCall):
            key = self._agg_key(expr)
            if key in agg_ctx:
                return agg_ctx[key]
            return None
        if isinstance(expr, BinaryOp):
            left = self._eval_having(expr.left, ctx, agg_ctx, row)
            if expr.op == "AND":
                return bool(left) and bool(self._eval_having(expr.right, ctx, agg_ctx, row))
            if expr.op == "OR":
                return bool(left) or bool(self._eval_having(expr.right, ctx, agg_ctx, row))
            right = self._eval_having(expr.right, ctx, agg_ctx, row)
            return self._eval_binop(expr.op, left, right)
        if isinstance(expr, UnaryOp):
            val = self._eval_having(expr.operand, ctx, agg_ctx, row)
            if expr.op == "NOT":
                return not val
            return val
        # Fall back to regular expression evaluation for non-aggregate parts
        return self._eval_expr(expr, ctx, row)

    def _extract_col_values(self, expr, rows, tdef):
        vals = []
        for row in rows:
            ctx = self._row_context(tdef, row) if tdef else {}
            vals.append(self._eval_expr(expr, ctx, row))
        return vals
 
    # heap scan
    def _scan_heap(self, tdef: TableDef, txn: Transaction):
        """Yield (RID, decoded_row) for all live records in a table's heap."""
        pid = tdef.heap_page
        while pid != INVALID_PAGE:
            page = self._pool.fetch_page(pid)
            for slot_idx, rec_bytes in page.iter_records():
                row = decode_record(tdef.columns, rec_bytes)
                yield RID(pid, slot_idx), row
            next_pid = page.overflow_pid
            self._pool.unpin(pid)
            pid = next_pid
 
    def _heap_insert(self, tdef: TableDef, record: bytes, txn: Transaction) -> RID:
        """Insert a record into the table's heap, allocating overflow pages as needed."""
        pid = tdef.heap_page
        while True:
            page = self._pool.fetch_page(pid)
            before = page.to_bytes()
            slot = page.insert(record)
            if slot is not None:
                after = page.to_bytes()
                lsn = self._txn.log_update(txn, pid, before, after)
                page.lsn = lsn
                page._write_header()
                self._pool.unpin(pid, dirty=True)
                return RID(pid, slot)
            next_pid = page.overflow_pid
            if next_pid == INVALID_PAGE:
                # allocate new page and chain it
                new_page = self._pool.new_page()
                new_page.page_type = PageType.DATA
                tdef.stats.page_count += 1
                page.overflow_pid = new_page.page_id
                page._write_header()
                after_old = page.to_bytes()
                lsn = self._txn.log_update(txn, pid, before, after_old)
                page.lsn = lsn
                page._write_header()
                self._pool.unpin(pid, dirty=True)
                before_new = new_page.to_bytes()
                slot = new_page.insert(record)
                after_new = new_page.to_bytes()
                npid = new_page.page_id
                lsn = self._txn.log_update(txn, npid, before_new, after_new)
                new_page.lsn = lsn
                new_page._write_header()
                self._pool.unpin(npid, dirty=True)
                return RID(npid, slot)
            self._pool.unpin(pid)
            pid = next_pid
 
    # expression evaluator 
    def _eval_expr(self, expr, ctx: dict, row: list) -> Any:
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, ColumnRef):
            name = expr.name.lower()
            if expr.table:
                key = f"{expr.table.lower()}.{name}"
                if key in ctx:
                    return ctx[key]
            if name in ctx:
                return ctx[name]
            # try positional
            try:
                return row[int(name)] if name.isdigit() else None
            except (IndexError, ValueError):
                return None
        if isinstance(expr, BinaryOp):
            left = self._eval_expr(expr.left, ctx, row)
            if expr.op == "AND":
                if not left:
                    return False
                return bool(self._eval_expr(expr.right, ctx, row))
            if expr.op == "OR":
                if left:
                    return True
                return bool(self._eval_expr(expr.right, ctx, row))
            right = self._eval_expr(expr.right, ctx, row)
            return self._eval_binop(expr.op, left, right)
        if isinstance(expr, UnaryOp):
            val = self._eval_expr(expr.operand, ctx, row)
            if expr.op == "NOT":
                return not val
            if expr.op == "-":
                return -val if val is not None else None
        if isinstance(expr, IsNullExpr):
            val = self._eval_expr(expr.expr, ctx, row)
            result = val is None
            return not result if expr.negated else result
        if isinstance(expr, InExpr):
            val = self._eval_expr(expr.expr, ctx, row)
            vals = [self._eval_expr(v, ctx, row) for v in expr.values]
            result = val in vals
            return not result if expr.negated else result
        if isinstance(expr, BetweenExpr):
            val = self._eval_expr(expr.expr, ctx, row)
            lo = self._eval_expr(expr.low, ctx, row)
            hi = self._eval_expr(expr.high, ctx, row)
            return lo <= val <= hi if val is not None else False
        if isinstance(expr, FuncCall):
            # aggregates are handled at the aggregate node level
            return None
        return None
 
    def _eval_binop(self, op: str, left, right):
        if left is None or right is None:
            return None if op not in ("=", "<>") else (
                (left is None and right is None) if op == "=" else
                not (left is None and right is None)
            )
        try:
            if op == "=":  return left == right
            if op == "<>": return left != right
            if op == "<":  return left < right
            if op == ">":  return left > right
            if op == "<=": return left <= right
            if op == ">=": return left >= right
            if op == "+":  return left + right
            if op == "-":  return left - right
            if op == "*":  return left * right
            if op == "/":  return left / right if right != 0 else None
            if op == "LIKE":
                pattern = str(right).replace("%", "*").replace("_", "?")
                return fnmatch.fnmatch(str(left), pattern)
        except TypeError:
            return None
        return None
 
    # helpers
    def _row_context(self, tdef: Optional[TableDef], row: list,
                     alias: Optional[str] = None) -> dict:
        if tdef is None:
            return {}
        ctx = {}
        for i, c in enumerate(tdef.columns):
            val = row[i] if i < len(row) else None
            ctx[c.name.lower()] = val
            ctx[f"{tdef.name.lower()}.{c.name.lower()}"] = val
            if alias:
                ctx[f"{alias.lower()}.{c.name.lower()}"] = val
        return ctx
 
    def _scan_tdef(self, node) -> Optional[TableDef]:
        """Walk plan tree to find the base table definition."""
        if isinstance(node, (SeqScan, IndexScan)):
            try:
                return self._cat.get_table(node.table)
            except KeyError:
                return None
        for attr in ("child", "left"):
            child = getattr(node, attr, None)
            if child:
                r = self._scan_tdef(child)
                if r:
                    return r
        return None
 
    def _infer_columns(self, node) -> list[str]:
        if isinstance(node, Projection):
            names = []
            for c in node.columns:
                if isinstance(c, ColumnRef):
                    names.append(c.name)
                elif isinstance(c, Literal) and c.value == "*":
                    tdef = self._scan_tdef(node)
                    return tdef.col_names if tdef else []
                elif isinstance(c, FuncCall):
                    args_str = ", ".join("*" if (isinstance(a, Literal) and a.value == "*")
                                         else (a.name if isinstance(a, ColumnRef) else "?")
                                         for a in c.args)
                    names.append(f"{c.name}({args_str})")
                else:
                    names.append("?")
            return names
        if isinstance(node, AggregateNode):
            names = []
            for g in node.group_by:
                if isinstance(g, ColumnRef):
                    names.append(g.name)
                else:
                    names.append("?")
            for a in node.aggregates:
                if isinstance(a, FuncCall):
                    args_str = ", ".join("*" if (isinstance(x, Literal) and x.value == "*")
                                         else (x.name if isinstance(x, ColumnRef) else "?")
                                         for x in a.args)
                    names.append(f"{a.name}({args_str})")
            return names if names else ["?"]
        # Default: all columns from the base table
        tdef = self._scan_tdef(node)
        if tdef:
            return tdef.col_names
        return []
    