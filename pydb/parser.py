"""
SQL Parser
==========

Overview
--------
This module transforms raw SQL text into a typed Abstract Syntax Tree
(AST) in two stages:

1. **Tokenizer** (``tokenize``) — splits the input string into a flat
   list of ``Token`` objects using regex-based scanning.  Each token
   carries a type (``TT`` enum), a value (the parsed literal or
   keyword), and a source position for error messages.

2. **Recursive-descent parser** (``Parser``) — consumes the token
   stream and builds a tree of AST dataclass nodes.  The parser
   implements operator-precedence climbing for expressions, giving
   correct handling of arithmetic, comparison, boolean, and
   grouping precedences.

The public entry point is ``parse_sql(sql) -> AST node``.

Supported SQL
~~~~~~~~~~~~~
* ``CREATE TABLE name (col type [PRIMARY KEY] [NOT NULL], ...)``
* ``DROP TABLE name``
* ``CREATE [UNIQUE] INDEX name ON table (col, ...)``
* ``INSERT INTO table (cols) VALUES (vals), ...``
* ``SELECT [DISTINCT] cols|* FROM table [alias]``
  ``[JOIN table ON cond]``
  ``[WHERE cond]``
  ``[GROUP BY cols [HAVING cond]]``
  ``[ORDER BY col [ASC|DESC], ...]``
  ``[LIMIT n [OFFSET m]]``
* ``UPDATE table SET col=val, ... [WHERE cond]``
* ``DELETE FROM table [WHERE cond]``
* ``BEGIN`` / ``COMMIT`` / ``ROLLBACK``
* ``EXPLAIN SELECT ...``

Expression precedence (low → high)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    OR
    AND
    NOT
    =  <>  <  >  <=  >=  LIKE  IS [NOT] NULL  IN  BETWEEN
    +  -
    *  /
    unary -
    primary (literal, column ref, function call, parenthesised expr)

AST nodes
~~~~~~~~~
Each SQL statement type has a corresponding dataclass:
``SelectStmt``, ``InsertStmt``, ``UpdateStmt``, ``DeleteStmt``,
``CreateTableStmt``, ``DropTableStmt``, ``CreateIndexStmt``,
``BeginStmt``, ``CommitStmt``, ``RollbackStmt``.

Expressions are represented as a tree of: ``ColumnRef``, ``Literal``,
``BinaryOp``, ``UnaryOp``, ``FuncCall``, ``IsNullExpr``, ``InExpr``,
``BetweenExpr``.

Error handling
~~~~~~~~~~~~~~
Parse errors raise ``ParseError`` with the unexpected token's type,
value, and source position, making it easy to report where in the
SQL string the error occurred.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Union

# ── Tokens ────────────────────────────────────────────────────────

class TT(Enum):
    """Token type enumeration.

    Covers three categories:

    * **Literals** — ``IDENT`` (identifiers), ``NUMBER`` (int/float),
      ``STRING`` (single-quoted).
    * **Keywords** — SQL reserved words (``SELECT`` through ``OFFSET``).
      During tokenization, identifiers are checked against the keyword
      map and promoted to their keyword token type.
    * **Symbols** — punctuation and operators (``LPAREN``, ``EQ``,
      ``PLUS``, etc.).
    * **EOF** — end of input sentinel.
    """
    # literals
    IDENT    = auto()
    NUMBER   = auto()
    STRING   = auto()
    # keywords (subset)
    SELECT   = auto(); FROM     = auto(); WHERE    = auto()
    INSERT   = auto(); INTO     = auto(); VALUES   = auto()
    UPDATE   = auto(); SET      = auto(); DELETE   = auto()
    CREATE   = auto(); DROP     = auto(); TABLE    = auto()
    INDEX    = auto(); ON       = auto(); UNIQUE   = auto()
    PRIMARY  = auto(); KEY      = auto(); NOT      = auto()
    NULL     = auto(); AND      = auto(); OR       = auto()
    ORDER    = auto(); BY       = auto(); ASC      = auto()
    DESC     = auto(); LIMIT    = auto(); JOIN     = auto()
    INNER    = auto(); LEFT     = auto(); RIGHT    = auto()
    CROSS    = auto(); AS       = auto(); BEGIN    = auto()
    COMMIT   = auto(); ROLLBACK = auto(); EXPLAIN  = auto()
    TRUE     = auto(); FALSE    = auto(); IS       = auto()
    IN       = auto(); LIKE     = auto(); BETWEEN  = auto()
    GROUP    = auto(); HAVING   = auto(); COUNT    = auto()
    SUM      = auto(); AVG      = auto(); MIN      = auto()
    MAX      = auto(); DISTINCT = auto(); OFFSET   = auto()
    # symbols
    LPAREN = auto(); RPAREN = auto(); COMMA   = auto()
    DOT    = auto(); SEMI   = auto(); STAR    = auto()
    EQ     = auto(); NE     = auto(); LT      = auto()
    GT     = auto(); LE     = auto(); GE      = auto()
    PLUS   = auto(); MINUS  = auto(); SLASH   = auto()
    # meta
    EOF    = auto()

_KEYWORDS = {k.name: k for k in TT if k.value >= TT.SELECT.value and k.value <= TT.OFFSET.value}

@dataclass
class Token:
    """A single lexical token produced by ``tokenize``.

    Attributes
    ----------
    tt : TT
        The token type.
    value : Any
        The parsed value — ``int`` or ``float`` for ``NUMBER``,
        ``str`` for ``STRING`` and ``IDENT``, raw string for symbols.
    pos : int
        Character offset in the original SQL string (for errors).
    """
    tt: TT
    value: Any
    pos: int

# ── Tokenizer ─────────────────────────────────────────────────────

_PATTERNS = [
    (r"--[^\n]*",              None),          # line comment
    (r"\s+",                   None),          # whitespace
    (r"'(?:''|[^'])*'",       "STRING"),
    (r"\d+\.\d+",             "FLOAT_LIT"),
    (r"\d+",                  "INT_LIT"),
    (r"<=",                   "LE"),
    (r">=",                   "GE"),
    (r"<>|!=",                "NE"),
    (r"=",                    "EQ"),
    (r"<",                    "LT"),
    (r">",                    "GT"),
    (r"\+",                   "PLUS"),
    (r"-",                    "MINUS"),
    (r"/",                    "SLASH"),
    (r"\*",                   "STAR"),
    (r"\(",                   "LPAREN"),
    (r"\)",                   "RPAREN"),
    (r",",                    "COMMA"),
    (r"\.",                   "DOT"),
    (r";",                    "SEMI"),
    (r"[A-Za-z_][A-Za-z0-9_]*", "IDENT"),
]
_RE = re.compile("|".join(f"(?P<G{i}>{p})" for i, (p, _) in enumerate(_PATTERNS)))

def tokenize(sql: str) -> list[Token]:
    """Scan a SQL string into a list of tokens.

    Uses a single compiled regex with named groups, one per pattern.
    Whitespace and line comments (``-- ...``) are discarded.
    Identifiers are checked against ``_KEYWORDS`` and promoted to
    their keyword token type if matched (case-insensitive).

    The returned list always ends with a ``TT.EOF`` sentinel so
    that the parser can safely peek without bounds checking.

    Parameters
    ----------
    sql : str
        The raw SQL input string.

    Returns
    -------
    list[Token]
        Tokens in source order, terminated by ``EOF``.
    """
    tokens: list[Token] = []
    for m in _RE.finditer(sql):
        idx = None
        for i in range(len(_PATTERNS)):
            if m.group(f"G{i}") is not None:
                idx = i
                break
        if idx is None:
            continue
        _, tag = _PATTERNS[idx]
        if tag is None:
            continue
        val = m.group()
        pos = m.start()

        if tag == "STRING":
            tokens.append(Token(TT.STRING, val[1:-1].replace("''", "'"), pos))
        elif tag == "FLOAT_LIT":
            tokens.append(Token(TT.NUMBER, float(val), pos))
        elif tag == "INT_LIT":
            tokens.append(Token(TT.NUMBER, int(val), pos))
        elif tag == "IDENT":
            kw = _KEYWORDS.get(val.upper())
            if kw:
                tokens.append(Token(kw, val, pos))
            else:
                tokens.append(Token(TT.IDENT, val, pos))
        else:
            tokens.append(Token(TT[tag], val, pos))

    tokens.append(Token(TT.EOF, None, len(sql)))
    return tokens

# ── AST nodes ─────────────────────────────────────────────────────

@dataclass
class ColumnRef:
    """Reference to a table column: ``[table.]name``."""
    table: Optional[str]
    name: str

@dataclass
class Literal:
    """A constant value: integer, float, string, bool, None, or ``'*'``."""
    value: Any

@dataclass
class BinaryOp:
    """Binary operator expression: ``left op right``.

    *op* is one of ``'+', '-', '*', '/', '=', '<>', '<', '>', '<=',
    '>=', 'AND', 'OR', 'LIKE'``.
    """
    op: str
    left: Any
    right: Any

@dataclass
class UnaryOp:
    """Unary operator expression: ``op operand`` (``NOT`` or ``-``)."""
    op: str
    operand: Any

@dataclass
class FuncCall:
    """Aggregate or scalar function call: ``name([DISTINCT] args)``."""
    name: str
    args: list
    distinct: bool = False

@dataclass
class IsNullExpr:
    """``expr IS [NOT] NULL`` expression."""
    expr: Any
    negated: bool = False

@dataclass
class InExpr:
    """``expr [NOT] IN (values)`` expression."""
    expr: Any
    values: list
    negated: bool = False

@dataclass
class BetweenExpr:
    """``expr BETWEEN low AND high`` expression."""
    expr: Any
    low: Any
    high: Any

@dataclass
class SelectStmt:
    """Parsed ``SELECT`` statement.

    Attributes
    ----------
    columns : list
        Selected expressions (``ColumnRef``, ``Literal('*')``, ``FuncCall``).
    from_table : TableRef or None
        The primary table (``None`` for expression-only SELECTs).
    joins : list[JoinClause]
        Zero or more JOIN clauses.
    where : AST node or None
        The WHERE predicate expression.
    order_by : list[tuple]
        ``[(expr, 'ASC'|'DESC'), ...]``.
    limit, offset : int or None
        Row count limits.
    group_by : list
        GROUP BY expressions.
    having : AST node or None
        The HAVING predicate (post-aggregation filter).
    explain : bool
        ``True`` if ``EXPLAIN`` was used.
    """
    columns: list              # [ColumnRef | Literal('*') | FuncCall]
    from_table: Optional["TableRef"] = None
    joins: list["JoinClause"] = field(default_factory=list)
    where: Any = None
    order_by: list[tuple] = field(default_factory=list)  # [(expr, 'ASC'|'DESC')]
    limit: Optional[int] = None
    offset: Optional[int] = None
    group_by: list = field(default_factory=list)
    having: Any = None
    explain: bool = False

@dataclass
class TableRef:
    """Table reference with optional alias: ``table [AS alias]``."""
    name: str
    alias: Optional[str] = None

@dataclass
class JoinClause:
    """A single JOIN clause within a SELECT statement."""
    join_type: str        # 'INNER','LEFT','RIGHT','CROSS'
    table: TableRef
    on: Any = None

@dataclass
class InsertStmt:
    """Parsed ``INSERT INTO table (cols) VALUES (...), ...`` statement."""
    table: str
    columns: list[str]
    rows: list[list]

@dataclass
class UpdateStmt:
    """Parsed ``UPDATE table SET col=val, ... [WHERE cond]`` statement."""
    table: str
    assignments: list[tuple[str, Any]]
    where: Any = None

@dataclass
class DeleteStmt:
    """Parsed ``DELETE FROM table [WHERE cond]`` statement."""
    table: str
    where: Any = None

@dataclass
class CreateTableStmt:
    """Parsed ``CREATE TABLE`` statement.

    ``columns`` is a list of ``(name, type_str, nullable, is_pk)`` tuples.
    """
    table: str
    columns: list[tuple[str, str, bool, bool]]

@dataclass
class DropTableStmt:
    """Parsed ``DROP TABLE name`` statement."""
    table: str

@dataclass
class CreateIndexStmt:
    """Parsed ``CREATE [UNIQUE] INDEX name ON table (cols)`` statement."""
    index_name: str
    table: str
    columns: list[str]
    unique: bool = False

@dataclass
class BeginStmt:
    """Parsed ``BEGIN`` statement."""
    pass
@dataclass
class CommitStmt:
    """Parsed ``COMMIT`` statement."""
    pass
@dataclass
class RollbackStmt:
    """Parsed ``ROLLBACK`` statement."""
    pass

class ParseError(Exception):
    pass

class Parser:
    """Recursive-descent SQL parser.

    Consumes a token list and builds an AST.  Each grammar rule is
    a method named after the production it handles (e.g. ``_select``,
    ``_expr``, ``_comparison``).  Expression parsing uses
    precedence climbing: ``_expr`` → ``_or_expr`` → ``_and_expr`` →
    ``_not_expr`` → ``_comparison`` → ``_addition`` → ``_multiplication``
    → ``_unary`` → ``_primary``.

    Parameters
    ----------
    tokens : list[Token]
        The token stream from ``tokenize()``, terminated by ``EOF``.
    """
    def __init__(self, tokens: list[Token]):
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        t = self._tokens[self._pos]
        self._pos += 1
        return t

    def _expect(self, tt: TT) -> Token:
        t = self._advance()
        if t.tt != tt:
            raise ParseError(f"Expected {tt.name} but got {t.tt.name} ('{t.value}') at pos {t.pos}")
        return t

    def _match(self, *tts: TT) -> Optional[Token]:
        if self._peek().tt in tts:
            return self._advance()
        return None

    # ── entry point ───────────────────────────────────────────────
    def parse(self):
        stmt = self._statement()
        self._match(TT.SEMI)
        return stmt

    def _statement(self):
        t = self._peek()
        if t.tt == TT.SELECT:
            return self._select()
        if t.tt == TT.EXPLAIN:
            self._advance()
            s = self._select()
            s.explain = True
            return s
        if t.tt == TT.INSERT:
            return self._insert()
        if t.tt == TT.UPDATE:
            return self._update()
        if t.tt == TT.DELETE:
            return self._delete()
        if t.tt == TT.CREATE:
            return self._create()
        if t.tt == TT.DROP:
            return self._drop()
        if t.tt == TT.BEGIN:
            self._advance(); return BeginStmt()
        if t.tt == TT.COMMIT:
            self._advance(); return CommitStmt()
        if t.tt == TT.ROLLBACK:
            self._advance(); return RollbackStmt()
        raise ParseError(f"Unexpected token {t.tt.name} at pos {t.pos}")

    # ── SELECT ────────────────────────────────────────────────────
    def _select(self):
        self._expect(TT.SELECT)
        cols = self._select_columns()
        from_table = None
        joins = []
        where = None
        group_by = []
        having = None
        order_by = []
        limit = None
        offset = None

        if self._match(TT.FROM):
            from_table = self._table_ref()
            while self._peek().tt in (TT.JOIN, TT.INNER, TT.LEFT, TT.RIGHT, TT.CROSS):
                joins.append(self._join_clause())
            if self._match(TT.WHERE):
                where = self._expr()
            if self._match(TT.GROUP):
                self._expect(TT.BY)
                group_by = [self._expr()]
                while self._match(TT.COMMA):
                    group_by.append(self._expr())
                if self._match(TT.HAVING):
                    having = self._expr()
            if self._match(TT.ORDER):
                self._expect(TT.BY)
                order_by = self._order_list()
            if self._match(TT.LIMIT):
                limit = int(self._expect(TT.NUMBER).value)
                if self._match(TT.OFFSET):
                    offset = int(self._expect(TT.NUMBER).value)

        return SelectStmt(cols, from_table, joins, where, order_by,
                          limit, offset, group_by, having)

    def _select_columns(self):
        if self._match(TT.STAR):
            return [Literal("*")]
        cols = [self._select_col()]
        while self._match(TT.COMMA):
            cols.append(self._select_col())
        return cols

    def _select_col(self):
        return self._expr()

    def _table_ref(self) -> TableRef:
        name = self._expect(TT.IDENT).value
        alias = None
        if self._match(TT.AS):
            alias = self._expect(TT.IDENT).value
        elif self._peek().tt == TT.IDENT and self._peek().tt not in (
                TT.WHERE, TT.ORDER, TT.JOIN, TT.INNER, TT.LEFT,
                TT.RIGHT, TT.CROSS, TT.GROUP, TT.LIMIT, TT.ON):
            alias = self._advance().value
        return TableRef(name, alias)

    def _join_clause(self) -> JoinClause:
        jt = "INNER"
        if self._match(TT.LEFT):
            jt = "LEFT"
        elif self._match(TT.RIGHT):
            jt = "RIGHT"
        elif self._match(TT.CROSS):
            jt = "CROSS"
        elif self._match(TT.INNER):
            pass
        self._expect(TT.JOIN)
        tbl = self._table_ref()
        on = None
        if self._match(TT.ON):
            on = self._expr()
        return JoinClause(jt, tbl, on)

    def _order_list(self):
        items = [self._order_item()]
        while self._match(TT.COMMA):
            items.append(self._order_item())
        return items

    def _order_item(self):
        e = self._expr()
        direction = "ASC"
        if self._match(TT.DESC):
            direction = "DESC"
        elif self._match(TT.ASC):
            pass
        return (e, direction)

    # ── expressions (precedence climbing) ─────────────────────────
    def _expr(self):
        return self._or_expr()

    def _or_expr(self):
        left = self._and_expr()
        while self._match(TT.OR):
            left = BinaryOp("OR", left, self._and_expr())
        return left

    def _and_expr(self):
        left = self._not_expr()
        while self._match(TT.AND):
            left = BinaryOp("AND", left, self._not_expr())
        return left

    def _not_expr(self):
        if self._match(TT.NOT):
            return UnaryOp("NOT", self._not_expr())
        return self._comparison()

    def _comparison(self):
        left = self._addition()
        if self._match(TT.EQ):
            return BinaryOp("=", left, self._addition())
        if self._match(TT.NE):
            return BinaryOp("<>", left, self._addition())
        if self._match(TT.LT):
            return BinaryOp("<", left, self._addition())
        if self._match(TT.GT):
            return BinaryOp(">", left, self._addition())
        if self._match(TT.LE):
            return BinaryOp("<=", left, self._addition())
        if self._match(TT.GE):
            return BinaryOp(">=", left, self._addition())
        if self._match(TT.LIKE):
            return BinaryOp("LIKE", left, self._addition())
        if self._peek().tt == TT.IS:
            self._advance()
            neg = bool(self._match(TT.NOT))
            self._expect(TT.NULL)
            return IsNullExpr(left, negated=neg)
        if self._peek().tt == TT.NOT:
            # lookahead for NOT IN / NOT BETWEEN
            saved = self._pos
            self._advance()
            if self._match(TT.IN):
                return self._in_list(left, negated=True)
            if self._match(TT.BETWEEN):
                return self._between(left, negated=True)
            self._pos = saved
        if self._match(TT.IN):
            return self._in_list(left)
        if self._match(TT.BETWEEN):
            return self._between(left)
        return left

    def _in_list(self, left, negated=False):
        self._expect(TT.LPAREN)
        vals = [self._expr()]
        while self._match(TT.COMMA):
            vals.append(self._expr())
        self._expect(TT.RPAREN)
        return InExpr(left, vals, negated)

    def _between(self, left, negated=False):
        lo = self._addition()
        self._expect(TT.AND)
        hi = self._addition()
        return BetweenExpr(left, lo, hi)

    def _addition(self):
        left = self._multiplication()
        while True:
            if self._match(TT.PLUS):
                left = BinaryOp("+", left, self._multiplication())
            elif self._match(TT.MINUS):
                left = BinaryOp("-", left, self._multiplication())
            else:
                break
        return left

    def _multiplication(self):
        left = self._unary()
        while True:
            if self._match(TT.STAR):
                left = BinaryOp("*", left, self._unary())
            elif self._match(TT.SLASH):
                left = BinaryOp("/", left, self._unary())
            else:
                break
        return left

    def _unary(self):
        if self._match(TT.MINUS):
            return UnaryOp("-", self._primary())
        return self._primary()

    def _primary(self):
        t = self._peek()
        # aggregate functions
        if t.tt in (TT.COUNT, TT.SUM, TT.AVG, TT.MIN, TT.MAX):
            return self._agg_func()
        if t.tt == TT.NUMBER:
            self._advance()
            return Literal(t.value)
        if t.tt == TT.STRING:
            self._advance()
            return Literal(t.value)
        if t.tt == TT.TRUE:
            self._advance()
            return Literal(True)
        if t.tt == TT.FALSE:
            self._advance()
            return Literal(False)
        if t.tt == TT.NULL:
            self._advance()
            return Literal(None)
        if t.tt == TT.LPAREN:
            self._advance()
            e = self._expr()
            self._expect(TT.RPAREN)
            return e
        if t.tt == TT.IDENT:
            self._advance()
            if self._match(TT.DOT):
                col = self._expect(TT.IDENT).value
                return ColumnRef(t.value, col)
            return ColumnRef(None, t.value)
        if t.tt == TT.STAR:
            self._advance()
            return Literal("*")
        raise ParseError(f"Unexpected {t.tt.name} at pos {t.pos}")

    def _agg_func(self):
        t = self._advance()
        self._expect(TT.LPAREN)
        distinct = bool(self._match(TT.DISTINCT))
        if self._match(TT.STAR):
            args = [Literal("*")]
        else:
            args = [self._expr()]
        self._expect(TT.RPAREN)
        return FuncCall(t.tt.name, args, distinct)

    # ── INSERT ────────────────────────────────────────────────────
    def _insert(self):
        self._expect(TT.INSERT)
        self._expect(TT.INTO)
        table = self._expect(TT.IDENT).value
        cols = []
        if self._match(TT.LPAREN):
            cols.append(self._expect(TT.IDENT).value)
            while self._match(TT.COMMA):
                cols.append(self._expect(TT.IDENT).value)
            self._expect(TT.RPAREN)
        self._expect(TT.VALUES)
        rows = [self._value_list()]
        while self._match(TT.COMMA):
            rows.append(self._value_list())
        return InsertStmt(table, cols, rows)

    def _value_list(self):
        self._expect(TT.LPAREN)
        vals = [self._expr()]
        while self._match(TT.COMMA):
            vals.append(self._expr())
        self._expect(TT.RPAREN)
        return vals

    # ── UPDATE ────────────────────────────────────────────────────
    def _update(self):
        self._expect(TT.UPDATE)
        table = self._expect(TT.IDENT).value
        self._expect(TT.SET)
        assignments = [self._assignment()]
        while self._match(TT.COMMA):
            assignments.append(self._assignment())
        where = None
        if self._match(TT.WHERE):
            where = self._expr()
        return UpdateStmt(table, assignments, where)

    def _assignment(self):
        col = self._expect(TT.IDENT).value
        self._expect(TT.EQ)
        val = self._expr()
        return (col, val)

    # ── DELETE ────────────────────────────────────────────────────
    def _delete(self):
        self._expect(TT.DELETE)
        self._expect(TT.FROM)
        table = self._expect(TT.IDENT).value
        where = None
        if self._match(TT.WHERE):
            where = self._expr()
        return DeleteStmt(table, where)

    # ── CREATE ────────────────────────────────────────────────────
    def _create(self):
        self._expect(TT.CREATE)
        if self._match(TT.TABLE):
            return self._create_table()
        unique = bool(self._match(TT.UNIQUE))
        self._expect(TT.INDEX)
        return self._create_index(unique)

    def _create_table(self):
        name = self._expect(TT.IDENT).value
        self._expect(TT.LPAREN)
        cols = [self._col_def()]
        while self._match(TT.COMMA):
            cols.append(self._col_def())
        self._expect(TT.RPAREN)
        return CreateTableStmt(name, cols)

    def _col_def(self):
        cname = self._expect(TT.IDENT).value
        ctype = self._expect(TT.IDENT).value
        nullable = True
        pk = False
        while self._peek().tt in (TT.PRIMARY, TT.NOT, TT.NULL):
            if self._match(TT.PRIMARY):
                self._expect(TT.KEY)
                pk = True
                nullable = False
            elif self._match(TT.NOT):
                self._expect(TT.NULL)
                nullable = False
        return (cname, ctype, nullable, pk)

    def _create_index(self, unique: bool):
        idx_name = self._expect(TT.IDENT).value
        self._expect(TT.ON)
        table = self._expect(TT.IDENT).value
        self._expect(TT.LPAREN)
        cols = [self._expect(TT.IDENT).value]
        while self._match(TT.COMMA):
            cols.append(self._expect(TT.IDENT).value)
        self._expect(TT.RPAREN)
        return CreateIndexStmt(idx_name, table, cols, unique)

    # ── DROP ──────────────────────────────────────────────────────
    def _drop(self):
        self._expect(TT.DROP)
        self._expect(TT.TABLE)
        name = self._expect(TT.IDENT).value
        return DropTableStmt(name)


def parse_sql(sql: str):
    """Parse a SQL string into an AST node.

    This is the public entry point for the parser module.  It
    tokenizes the input and runs the recursive-descent parser.

    Parameters
    ----------
    sql : str
        A single SQL statement.

    Returns
    -------
    AST node
        One of ``SelectStmt``, ``InsertStmt``, ``UpdateStmt``,
        ``DeleteStmt``, ``CreateTableStmt``, ``DropTableStmt``,
        ``CreateIndexStmt``, ``BeginStmt``, ``CommitStmt``,
        ``RollbackStmt``.

    Raises
    ------
    ParseError
        If the SQL is syntactically invalid.
    """
    tokens = tokenize(sql)
    return Parser(tokens).parse()
