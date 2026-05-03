"""
Query Planner
=============
 
Overview
--------
The planner translates an AST (from the parser) into a **physical
query plan** — a tree of plan-node dataclasses that the executor
walks to produce results.
 
Planning steps for a ``SELECT``:
 
1. **Resolve access paths** — for the base table, choose between a
   ``SeqScan`` (full table scan) and an ``IndexScan`` (B+Tree
   lookup).  The decision is cost-based: we extract equality
   predicates (``col = literal``) from the ``WHERE`` clause and
   check if any index's leading columns match.
2. **Build joins** — each ``JOIN`` becomes a ``NestedLoopJoin``
   node pairing the running plan with a scan of the joined table.
3. **Insert aggregation** — if the query has aggregate functions
   (``COUNT``, ``SUM``, etc.) or a ``GROUP BY``, an
   ``AggregateNode`` is inserted.
4. **Insert sort** — if ``ORDER BY`` is present and no index
   naturally produces the requested order, a ``SortNode`` is
   added (which may invoke the external merge sorter at runtime).
5. **Insert projection** — a ``Projection`` node extracts or
   computes the requested columns.
6. **Insert limit** — ``LimitNode`` caps the output.
 
For DML statements (``INSERT``, ``UPDATE``, ``DELETE``), the planner
wraps the scan/filter logic inside the appropriate plan node and
lets the executor handle the mutation.
 
Plan node types
~~~~~~~~~~~~~~~
* ``SeqScan``          — full table scan with optional pushed-down filter.
* ``IndexScan``        — B+Tree lookup with key and residual filter.
* ``NestedLoopJoin``   — standard nested-loop join (inner/left/right/cross).
* ``Filter``           — standalone predicate filter.
* ``Projection``       — column selection / expression evaluation.
* ``SortNode``         — ORDER BY (in-memory or external merge sort).
* ``LimitNode``        — LIMIT + OFFSET.
* ``AggregateNode``    — GROUP BY + aggregate functions + HAVING.
* ``InsertPlan``       — INSERT INTO ... VALUES ...
* ``UpdatePlan``       — UPDATE ... SET ... WHERE ...
* ``DeletePlan``       — DELETE FROM ... WHERE ...
* ``CreateTablePlan``  — CREATE TABLE DDL.
* ``DropTablePlan``    — DROP TABLE DDL.
* ``CreateIndexPlan``  — CREATE INDEX DDL.
* ``BeginPlan`` / ``CommitPlan`` / ``RollbackPlan`` — transaction control.
* ``ExplainPlan``      — wraps another plan for EXPLAIN output.
 
The ``format_plan`` function pretty-prints a plan tree as an indented
string for ``EXPLAIN`` output.
"""

from __future__ import annotations
 
from dataclasses import dataclass
from typing import Any, Optional

from pydb.catalog import Catalog, TableDef, IndexDef
from pydb.parser import (
    SelectStmt, InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt, CreateIndexStmt,
    BeginStmt, CommitStmt, RollbackStmt,
    CreateUserStmt, DropUserStmt, AlterUserStmt,
    ColumnRef, Literal, BinaryOp, FuncCall,
)

@dataclass
class SeqScan:
    """Full sequential scan of a table's heap pages.
 
    Attributes
    ----------
    table : str
        Table name.
    alias : str or None
        SQL alias (e.g. ``FROM users u`` → alias is ``"u"``).
    filter : AST node or None
        WHERE predicate pushed down into the scan.
    """
    table: str
    alias: Optional[str] = None
    filter: Any = None
    
@dataclass
class IndexScan:
    """B+Tree index lookup.
 
    Attributes
    ----------
    table : str
        Table name.
    index_name : str
        Which index to use.
    alias : str or None
        SQL alias.
    lookup_key : dict or None
        Equality predicates matched to leading index columns,
        as ``{col_name: literal_value}``.
    filter : AST node or None
        Residual predicate applied after the index lookup.
    """
    table: str
    index_name: str
    alias: Optional[str] = None
    lookup_key: Any = None
    filter: Any = None
    
@dataclass
class NestedLoopJoin:
    """Nested-loop join: for each row in *left*, scan all of *right*."""
    left: Any
    right: Any
    join_type: str = "INNER"
    on: Any = None
    
@dataclass
class Filter:
    """Standalone predicate filter over a child plan."""
    child: Any
    predicate: Any
 
@dataclass
class Projection:
    """Column selection / expression evaluation."""
    child: Any
    columns: list
 
@dataclass
class SortNode:
    """ORDER BY — delegates to in-memory sort or external merge sort."""
    child: Any
    order_by: list[tuple]  # [(expr, 'ASC'|'DESC')]
    
@dataclass
class LimitNode:
    """Row-count cap with optional offset."""
    child: Any
    limit: int
    offset: int = 0
 
@dataclass
class AggregateNode:
    """GROUP BY + aggregate functions (COUNT, SUM, AVG, MIN, MAX)."""
    child: Any
    group_by: list
    aggregates: list
    having: Any = None
    
@dataclass
class InsertPlan:
    """Physical plan for INSERT statements."""
    table: str
    columns: list[str]
    rows: list[list]
    
@dataclass
class UpdatePlan:
    """Physical plan for UPDATE statements."""
    table: str
    assignments: list[tuple]
    scan: Any
 
@dataclass
class DeletePlan:
    """Physical plan for DELETE statements."""
    table: str
    scan: Any
 
@dataclass
class CreateTablePlan:
    """Physical plan for CREATE TABLE DDL."""
    stmt: CreateTableStmt
 
@dataclass
class DropTablePlan:
    """Physical plan for DROP TABLE DDL."""
    table: str
 
@dataclass
class CreateIndexPlan:
    """Physical plan for CREATE INDEX DDL."""
    stmt: CreateIndexStmt
 
@dataclass
class BeginPlan:
    """Physical plan for BEGIN transaction."""
    pass

@dataclass
class CommitPlan:
    """Physical plan for COMMIT transaction."""
    pass
@dataclass
class RollbackPlan:
    """Physical plan for ROLLBACK transaction."""
    pass
 
@dataclass
class ExplainPlan:
    """Wraps another plan — executor prints the plan tree instead of running it."""
    child: Any

@dataclass
class CreateUserPlan:
    """Physical plan for CREATE USER."""
    username: str
    password: str

@dataclass
class DropUserPlan:
    """Physical plan for DROP USER."""
    username: str

@dataclass
class AlterUserPlan:
    """Physical plan for ALTER USER ... SET PASSWORD."""
    username: str
    new_password: str
    
class Planner:
    """Cost-based query planner.
 
    Translates parsed AST nodes into physical plan trees.  The main
    optimisation is **index selection**: for ``SELECT`` queries with
    equality predicates in ``WHERE``, the planner checks whether any
    index's leading columns match those predicates and chooses an
    ``IndexScan`` over a ``SeqScan`` when possible.
 
    Parameters
    ----------
    catalog : Catalog
        The system catalog, used to look up table schemas and
        available indexes.
    """
    
    def __init__(self, catalog: Catalog):
        self._cat = catalog
        
    def plan(self, stmt) -> Any:
        if isinstance(stmt, SelectStmt):
            return self._plan_select(stmt)
        if isinstance(stmt, InsertStmt):
            return InsertPlan(stmt.table, stmt.columns, stmt.rows)
        if isinstance(stmt, UpdateStmt):
            scan = self._make_scan(stmt.table, stmt.where)
            return UpdatePlan(stmt.table, stmt.assignments, scan)
        if isinstance(stmt, DeleteStmt):
            scan = self._make_scan(stmt.table, stmt.where)
            return DeletePlan(stmt.table, scan)
        if isinstance(stmt, CreateTableStmt):
            return CreateTablePlan(stmt)
        if isinstance(stmt, DropTableStmt):
            return DropTablePlan(stmt.table)
        if isinstance(stmt, CreateIndexStmt):
            return CreateIndexPlan(stmt)
        if isinstance(stmt, BeginStmt):
            return BeginPlan()
        if isinstance(stmt, CommitStmt):
            return CommitPlan()
        if isinstance(stmt, RollbackStmt):
            return RollbackPlan()
        if isinstance(stmt, CreateUserStmt):
            return CreateUserPlan(stmt.username, stmt.password)
        if isinstance(stmt, DropUserStmt):
            return DropUserPlan(stmt.username)
        if isinstance(stmt, AlterUserStmt):
            return AlterUserPlan(stmt.username, stmt.new_password)
        raise ValueError(f"Unknown statement type: {type(stmt)}")
    
    def _plan_select(self, stmt: SelectStmt):
        if stmt.from_table is None:
            # SELECT 1 + 1, etc. - no table scan needed
            plan = Projection(None, stmt.columns)
            if stmt.explain:
                return ExplainPlan(plan)
            return plan
        
        # base scan
        plan = self._make_scan(stmt.from_table.name, stmt.where,
                               alias=stmt.from_table.alias)
        
        # joins 
        for j in stmt.joins:
            right = self._make_scan(j.table.name, None, alias=j.table.alias)
            plan = NestedLoopJoin(plan, right, j.join_type, j.on)
            
        # aggregation
        aggs = [c for c in stmt.columns if isinstance(c, FuncCall)]
        if stmt.group_by or aggs:
            plan = AggregateNode(plan, stmt.group_by, aggs, stmt.having)
            # add projection to ensure correct column order
            plan = Projection(plan, stmt.columns)
            # limit / offset
            if stmt.limit is not None:
                plan = LimitNode(plan, stmt.limit, stmt.offset or 0)
            if stmt.explain:
                return ExplainPlan(plan)
            return plan
        
        # order by — BEFORE projection so sort can see all columns
        if stmt.order_by:
            if not self._order_covered_by_index(stmt):
                plan = SortNode(plan, stmt.order_by)
        
        # projection
        is_star = any(isinstance(c, Literal) and c.value == "*" for c in stmt.columns)
        if not is_star:
            plan = Projection(plan, stmt.columns)
 
        # limit / offset
        if stmt.limit is not None:
            plan = LimitNode(plan, stmt.limit, stmt.offset or 0)
 
        if stmt.explain:
            return ExplainPlan(plan)
        return plan
    
    def _make_scan(self, table: str, where: Any,
                   alias: Optional[str] = None):
        """Choose between SeqScan and IndexScan based on cost heuristic."""
        try:
            tdef = self._cat.get_table(table)
        except KeyError:
            return SeqScan(table, alias, where)
        
        # attempt to find a usable index for equality predicates
        idx = self._find_best_index(tdef, where)
        if idx is not None:
            idx_def, lookup = idx
            return IndexScan(table, idx_def.name, alias, lookup, where)
        
        return SeqScan(table, alias, where)
    
    def _find_best_index(self, tdef: TableDef, where: Any) -> Optional[tuple[IndexDef, Any]]:
        """Look for equality predicates that match an index prefix."""
        if where is None:
            return None
        eq_cols = self._extract_eq_columns(where)
        if not eq_cols:
            return None
        
        best: Optional[tuple[IndexDef, Any]] = None
        best_score = 0
        for idx in tdef.indexes.values():
            # count how many leading index columns have equality predicates
            score = 0
            for ic in idx.columns:
                if ic.lower() in eq_cols:
                    score += 1
                else:
                    break
            if score > best_score:
                best_score = score
                lookup = {ic.lower(): eq_cols[ic.lower()] for ic in idx.columns[:score]}
                best = (idx, lookup)
        return best
    
    def _extract_eq_columns(self, expr) -> dict[str, Any]:
        """Extract col=literal equalities from a WHERE clause."""
        result: dict[str, Any] = {}
        if isinstance(expr, BinaryOp):
            if expr.op == "AND":
                result.update(self._extract_eq_columns(expr.left))
                result.update(self._extract_eq_columns(expr.right))
            elif expr.op == "=":
                if isinstance(expr.left, ColumnRef) and isinstance(expr.right, Literal):
                    result[expr.left.name.lower()] = expr.right.value
                elif isinstance(expr.right, ColumnRef) and isinstance(expr.left, Literal):
                    result[expr.right.name.lower()] = expr.left.value
        return result
    
    def _order_covered_by_index(self, stmt: SelectStmt) -> bool:
        """Check if ORDER BY is naturally satisfied by an index."""
        if not stmt.order_by or not stmt.from_table:
            return False
        try:
            tdef = self._cat.get_table(stmt.from_table.name)
        except KeyError:
            return False
        order_cols = []
        for expr, _ in stmt.order_by:
            if isinstance(expr, ColumnRef):
                order_cols.append(expr.name.lower())
            else:
                return False
        for idx in tdef.indexes.values():
            idx_cols = [c.lower() for c in idx.columns]
            if idx_cols[:len(order_cols)] == order_cols:
                return True
        return False
    
def format_plan(node, indent: int = 0) -> str:
    """Pretty-print a query plan tree as an indented string.
 
    Used by ``EXPLAIN`` to display the physical plan the planner
    chose.  Each node type is printed with its key attributes
    (table name, index name, filter predicate), and children are
    indented by two spaces per level.
 
    Parameters
    ----------
    node : plan node
        The root of the plan tree to format.
    indent : int
        Current indentation level (used for recursion).
 
    Returns
    -------
    str
        Multi-line string representation of the plan tree.
    """
    prefix = "  " * indent
    lines = []
    if isinstance(node, SeqScan):
        f = f" filter={_fmt_expr(node.filter)}" if node.filter else ""
        lines.append(f"{prefix}SeqScan({node.table}{f})")
    elif isinstance(node, IndexScan):
        lines.append(f"{prefix}IndexScan({node.table} using {node.index_name})")
    elif isinstance(node, NestedLoopJoin):
        lines.append(f"{prefix}NestedLoopJoin({node.join_type})")
        lines.append(format_plan(node.left, indent + 1))
        lines.append(format_plan(node.right, indent + 1))
    elif isinstance(node, Filter):
        lines.append(f"{prefix}Filter({_fmt_expr(node.predicate)})")
        lines.append(format_plan(node.child, indent + 1))
    elif isinstance(node, Projection):
        lines.append(f"{prefix}Projection")
        if node.child:
            lines.append(format_plan(node.child, indent + 1))
    elif isinstance(node, SortNode):
        lines.append(f"{prefix}Sort")
        lines.append(format_plan(node.child, indent + 1))
    elif isinstance(node, LimitNode):
        lines.append(f"{prefix}Limit({node.limit} offset={node.offset})")
        lines.append(format_plan(node.child, indent + 1))
    elif isinstance(node, AggregateNode):
        lines.append(f"{prefix}Aggregate")
        lines.append(format_plan(node.child, indent + 1))
    elif isinstance(node, ExplainPlan):
        lines.append(format_plan(node.child, indent))
    else:
        lines.append(f"{prefix}{type(node).__name__}")
    return "\n".join(lines)
 
 
def _fmt_expr(expr) -> str:
    if expr is None:
        return "None"
    if isinstance(expr, ColumnRef):
        return f"{expr.table + '.' if expr.table else ''}{expr.name}"
    if isinstance(expr, Literal):
        return repr(expr.value)
    if isinstance(expr, BinaryOp):
        return f"({_fmt_expr(expr.left)} {expr.op} {_fmt_expr(expr.right)})"
    if isinstance(expr, FuncCall):
        return f"{expr.name}({', '.join(_fmt_expr(a) for a in expr.args)})"
    return str(expr)
