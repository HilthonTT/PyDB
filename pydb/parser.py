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
   implements operator-ecedence climbing for expressions, giving
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


# ---------------------------------------------------------------------------
# ParseError
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when the parser encounters unexpected input."""

    def __init__(self, message: str, token: Optional[Token] = None):
        self.token = token
        pos_info = f" at position {token.pos}" if token else ""
        super().__init__(f"{message}{pos_info}")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<WS>      \s+                    )
  | (?P<STRING>  '(?:''|[^'])*'         )
  | (?P<NUMBER>  \d+(?:\.\d+)?          )
  | (?P<IDENT>   [A-Za-z_]\w*           )
  | (?P<NE>      <>                     )
  | (?P<LE>      <=                     )
  | (?P<GE>      >=                     )
  | (?P<LPAREN>  \(                     )
  | (?P<RPAREN>  \)                     )
  | (?P<COMMA>   ,                      )
  | (?P<DOT>     \.                     )
  | (?P<SEMI>    ;                      )
  | (?P<STAR>    \*                     )
  | (?P<EQ>      =                      )
  | (?P<LT>      <                      )
  | (?P<GT>      >                      )
  | (?P<PLUS>    \+                     )
  | (?P<MINUS>   -                      )
  | (?P<SLASH>   /                      )
    """,
    re.VERBOSE,
)


def tokenize(sql: str) -> list[Token]:
    """Split a SQL string into a list of ``Token`` objects.

    Identifiers are checked against ``_KEYWORDS`` and promoted to
    their keyword token type when matched.  A sentinel ``EOF`` token
    is appended at the end.
    """
    tokens: list[Token] = []
    expected = 0

    for m in _TOKEN_RE.finditer(sql):
        if m.start() != expected:
            raise ParseError(f"Unexpected character {sql[expected]!r}",
                             Token(TT.EOF, None, expected))
        expected = m.end()
        kind = m.lastgroup
        text = m.group()
        pos = m.start()

        if kind == "WS":
            continue
        elif kind == "STRING":
            tokens.append(Token(TT.STRING, text[1:-1].replace("''", "'"), pos))
        elif kind == "NUMBER":
            val: Union[int, float] = float(text) if "." in text else int(text)
            tokens.append(Token(TT.NUMBER, val, pos))
        elif kind == "IDENT":
            upper = text.upper()
            tt = _KEYWORDS.get(upper, TT.IDENT)
            tokens.append(Token(tt, text, pos))
        else:
            tokens.append(Token(TT[kind], text, pos))

    if expected != len(sql):
        raise ParseError(f"Unexpected character {sql[expected]!r}",
                         Token(TT.EOF, None, expected))

    tokens.append(Token(TT.EOF, None, len(sql)))
    return tokens


# ---------------------------------------------------------------------------
# AST — Expression Nodes
# ---------------------------------------------------------------------------

@dataclass
class ColumnRef:
    """Column reference, optionally table-qualified.  ``name="*"`` for wildcards."""
    table: Optional[str]
    name: str

@dataclass
class Literal:
    """Literal value: int, float, str, bool, or None (SQL NULL)."""
    value: Any

@dataclass
class BinaryOp:
    """Binary operation (arithmetic, comparison, or boolean)."""
    op: str
    left: Any
    right: Any

@dataclass
class UnaryOp:
    """Unary NOT or unary minus."""
    op: str
    operand: Any

@dataclass
class FuncCall:
    """Function call, including aggregates (COUNT, SUM, etc.)."""
    name: str
    args: list
    distinct: bool = False

@dataclass
class IsNullExpr:
    """IS NULL / IS NOT NULL test."""
    expr: Any
    negated: bool

@dataclass
class InExpr:
    """IN (value_list) / NOT IN (value_list)."""
    expr: Any
    values: list
    negated: bool = False

@dataclass
class BetweenExpr:
    """BETWEEN low AND high / NOT BETWEEN low AND high."""
    expr: Any
    low: Any
    high: Any
    negated: bool = False


# ---------------------------------------------------------------------------
# AST — Helper Nodes
# ---------------------------------------------------------------------------

@dataclass
class SelectColumn:
    """One item in a SELECT column list."""
    expr: Any
    alias: Optional[str]

@dataclass
class OrderByItem:
    """One ORDER BY element."""
    expr: Any
    descending: bool

@dataclass
class JoinClause:
    """A single JOIN clause."""
    join_type: str          # INNER, LEFT, RIGHT, CROSS
    table: str
    alias: Optional[str]
    condition: Any          # ON expression; None for CROSS JOIN

@dataclass
class ColumnDef:
    """Column definition in CREATE TABLE (parser-level, not catalog-level)."""
    name: str
    type_name: str          # raw SQL type string, e.g. "VARCHAR(255)"
    primary_key: bool = False
    nullable: bool = True


# ---------------------------------------------------------------------------
# AST — Statement Nodes
# ---------------------------------------------------------------------------

@dataclass
class SelectStmt:
    columns: list           # list[SelectColumn]
    from_table: Optional[str]
    from_alias: Optional[str]
    joins: list             # list[JoinClause]
    where: Any
    group_by: list
    having: Any
    order_by: list          # list[OrderByItem]
    limit: Any
    offset: Any
    distinct: bool = False

@dataclass
class InsertStmt:
    table: str
    columns: list           # list[str]
    values: list            # list[list] — each inner list is one row of exprs

@dataclass
class UpdateStmt:
    table: str
    assignments: list       # list[tuple[str, expr]]
    where: Any

@dataclass
class DeleteStmt:
    table: str
    where: Any

@dataclass
class CreateTableStmt:
    name: str
    columns: list           # list[ColumnDef]

@dataclass
class DropTableStmt:
    name: str

@dataclass
class CreateIndexStmt:
    name: str
    table: str
    columns: list           # list[str]
    unique: bool

@dataclass
class BeginStmt:
    pass

@dataclass
class CommitStmt:
    pass

@dataclass
class RollbackStmt:
    pass

@dataclass
class ExplainStmt:
    stmt: Any


# ---------------------------------------------------------------------------
# Keywords that start SQL clauses (used to stop implicit alias detection)
# ---------------------------------------------------------------------------

_CLAUSE_KW = frozenset({
    TT.FROM, TT.WHERE, TT.GROUP, TT.ORDER, TT.LIMIT, TT.OFFSET,
    TT.JOIN, TT.INNER, TT.LEFT, TT.RIGHT, TT.CROSS,
    TT.ON, TT.HAVING, TT.SET, TT.VALUES, TT.INTO,
})

# Aggregate function token types
_AGG_FUNCS = frozenset({TT.COUNT, TT.SUM, TT.AVG, TT.MIN, TT.MAX})


# ---------------------------------------------------------------------------
# Recursive-Descent Parser
# ---------------------------------------------------------------------------

class Parser:
    """Recursive-descent parser with Pratt-style precedence climbing."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # -- infrastructure ----------------------------------------------------

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, tt: TT) -> Token:
        tok = self._peek()
        if tok.tt != tt:
            raise ParseError(f"Expected {tt.name}, got {tok.tt.name} ({tok.value!r})", tok)
        return self._advance()

    def _match(self, *tts: TT) -> Optional[Token]:
        if self._peek().tt in tts:
            return self._advance()
        return None

    def _at(self, *tts: TT) -> bool:
        return self._peek().tt in tts

    # -- top-level dispatch ------------------------------------------------

    def parse(self):
        tok = self._peek()
        if tok.tt == TT.SELECT:
            stmt = self._parse_select()
        elif tok.tt == TT.INSERT:
            stmt = self._parse_insert()
        elif tok.tt == TT.UPDATE:
            stmt = self._parse_update()
        elif tok.tt == TT.DELETE:
            stmt = self._parse_delete()
        elif tok.tt == TT.CREATE:
            stmt = self._parse_create()
        elif tok.tt == TT.DROP:
            stmt = self._parse_drop()
        elif tok.tt == TT.BEGIN:
            self._advance()
            stmt = BeginStmt()
        elif tok.tt == TT.COMMIT:
            self._advance()
            stmt = CommitStmt()
        elif tok.tt == TT.ROLLBACK:
            self._advance()
            stmt = RollbackStmt()
        elif tok.tt == TT.EXPLAIN:
            stmt = self._parse_explain()
        else:
            raise ParseError(f"Unexpected token {tok.tt.name}", tok)

        self._match(TT.SEMI)
        if not self._at(TT.EOF):
            t = self._peek()
            raise ParseError(f"Expected end of input, got {t.tt.name}", t)
        return stmt

    # -- statement parsers -------------------------------------------------

    def _parse_select(self) -> SelectStmt:
        self._expect(TT.SELECT)
        distinct = self._match(TT.DISTINCT) is not None
        columns = self._parse_select_columns()
        self._expect(TT.FROM)
        from_table = self._parse_table_name()
        from_alias = self._parse_optional_alias()
        joins = self._parse_joins()
        where = self._parse_expression() if self._match(TT.WHERE) else None
        group_by: list = []
        having = None
        if self._match(TT.GROUP):
            self._expect(TT.BY)
            group_by = self._parse_expression_list()
            if self._match(TT.HAVING):
                having = self._parse_expression()
        order_by = self._parse_order_by() if self._at(TT.ORDER) else []
        limit = self._parse_expression() if self._match(TT.LIMIT) else None
        offset = self._parse_expression() if self._match(TT.OFFSET) else None
        return SelectStmt(columns, from_table, from_alias, joins, where,
                          group_by, having, order_by, limit, offset, distinct)

    def _parse_table_name(self) -> str:
        tok = self._peek()
        if tok.tt == TT.IDENT:
            return self._advance().value
        # Allow keywords used as table names in some contexts
        raise ParseError(f"Expected table name, got {tok.tt.name}", tok)

    def _parse_optional_alias(self) -> Optional[str]:
        if self._match(TT.AS):
            return self._expect(TT.IDENT).value
        if self._at(TT.IDENT) and self._peek().tt not in _CLAUSE_KW:
            return self._advance().value
        return None

    def _parse_select_columns(self) -> list:
        # Bare * for SELECT *
        if self._at(TT.STAR):
            self._advance()
            return [SelectColumn(ColumnRef(None, "*"), None)]

        cols: list[SelectColumn] = []
        while True:
            expr = self._parse_expression()
            alias: Optional[str] = None
            if self._match(TT.AS):
                alias = self._expect(TT.IDENT).value
            elif self._at(TT.IDENT) and self._peek().tt not in _CLAUSE_KW:
                alias = self._advance().value
            cols.append(SelectColumn(expr, alias))
            if not self._match(TT.COMMA):
                break
        return cols

    def _parse_joins(self) -> list:
        joins: list[JoinClause] = []
        while self._at(TT.JOIN, TT.INNER, TT.LEFT, TT.RIGHT, TT.CROSS):
            join_type = "INNER"
            tok = self._peek()
            if tok.tt in (TT.INNER, TT.LEFT, TT.RIGHT, TT.CROSS):
                join_type = tok.tt.name
                self._advance()
            self._expect(TT.JOIN)
            table = self._parse_table_name()
            alias = self._parse_optional_alias()
            condition = None
            if join_type != "CROSS":
                self._expect(TT.ON)
                condition = self._parse_expression()
            joins.append(JoinClause(join_type, table, alias, condition))
        return joins

    def _parse_order_by(self) -> list:
        self._expect(TT.ORDER)
        self._expect(TT.BY)
        items: list[OrderByItem] = []
        while True:
            expr = self._parse_expression()
            desc = False
            if self._match(TT.DESC):
                desc = True
            else:
                self._match(TT.ASC)
            items.append(OrderByItem(expr, desc))
            if not self._match(TT.COMMA):
                break
        return items

    def _parse_insert(self) -> InsertStmt:
        self._expect(TT.INSERT)
        self._expect(TT.INTO)
        table = self._parse_table_name()
        self._expect(TT.LPAREN)
        columns = self._parse_ident_list()
        self._expect(TT.RPAREN)
        self._expect(TT.VALUES)
        rows: list[list] = []
        while True:
            self._expect(TT.LPAREN)
            row = self._parse_expression_list()
            self._expect(TT.RPAREN)
            rows.append(row)
            if not self._match(TT.COMMA):
                break
        return InsertStmt(table, columns, rows)

    def _parse_update(self) -> UpdateStmt:
        self._expect(TT.UPDATE)
        table = self._parse_table_name()
        self._expect(TT.SET)
        assignments: list[tuple] = []
        while True:
            col = self._expect(TT.IDENT).value
            self._expect(TT.EQ)
            val = self._parse_expression()
            assignments.append((col, val))
            if not self._match(TT.COMMA):
                break
        where = self._parse_expression() if self._match(TT.WHERE) else None
        return UpdateStmt(table, assignments, where)

    def _parse_delete(self) -> DeleteStmt:
        self._expect(TT.DELETE)
        self._expect(TT.FROM)
        table = self._parse_table_name()
        where = self._parse_expression() if self._match(TT.WHERE) else None
        return DeleteStmt(table, where)

    def _parse_create(self):
        self._expect(TT.CREATE)
        unique = self._match(TT.UNIQUE) is not None
        if self._at(TT.TABLE):
            if unique:
                raise ParseError("UNIQUE not valid for CREATE TABLE", self._peek())
            return self._parse_create_table()
        if self._at(TT.INDEX):
            return self._parse_create_index(unique)
        raise ParseError(f"Expected TABLE or INDEX after CREATE", self._peek())

    def _parse_create_table(self) -> CreateTableStmt:
        self._expect(TT.TABLE)
        name = self._parse_table_name()
        self._expect(TT.LPAREN)
        cols: list[ColumnDef] = []
        while True:
            cols.append(self._parse_column_def())
            if not self._match(TT.COMMA):
                break
        self._expect(TT.RPAREN)
        return CreateTableStmt(name, cols)

    def _parse_column_def(self) -> ColumnDef:
        name = self._expect(TT.IDENT).value
        # Type name — may be a keyword like INT, or an ident like VARCHAR
        type_tok = self._peek()
        if type_tok.tt == TT.IDENT or type_tok.tt in _KEYWORDS.values():
            type_name = self._advance().value
        else:
            raise ParseError(f"Expected column type, got {type_tok.tt.name}", type_tok)
        # Handle parenthesized suffix like VARCHAR(255)
        if self._match(TT.LPAREN):
            inner = self._expect(TT.NUMBER).value
            self._expect(TT.RPAREN)
            type_name = f"{type_name}({inner})"
        pk = False
        nullable = True
        # Parse constraints (any order, any number)
        while True:
            if self._at(TT.PRIMARY):
                self._advance()
                self._expect(TT.KEY)
                pk = True
                nullable = False
            elif self._at(TT.NOT):
                self._advance()
                self._expect(TT.NULL)
                nullable = False
            else:
                break
        return ColumnDef(name, type_name, pk, nullable)

    def _parse_create_index(self, unique: bool) -> CreateIndexStmt:
        self._expect(TT.INDEX)
        name = self._expect(TT.IDENT).value
        self._expect(TT.ON)
        table = self._parse_table_name()
        self._expect(TT.LPAREN)
        columns = self._parse_ident_list()
        self._expect(TT.RPAREN)
        return CreateIndexStmt(name, table, columns, unique)

    def _parse_drop(self) -> DropTableStmt:
        self._expect(TT.DROP)
        self._expect(TT.TABLE)
        name = self._parse_table_name()
        return DropTableStmt(name)

    def _parse_explain(self) -> ExplainStmt:
        self._expect(TT.EXPLAIN)
        stmt = self._parse_select()
        return ExplainStmt(stmt)

    # -- helpers -----------------------------------------------------------

    def _parse_ident_list(self) -> list[str]:
        names: list[str] = []
        while True:
            names.append(self._expect(TT.IDENT).value)
            if not self._match(TT.COMMA):
                break
        return names

    def _parse_expression_list(self) -> list:
        exprs: list = []
        while True:
            exprs.append(self._parse_expression())
            if not self._match(TT.COMMA):
                break
        return exprs

    # -- expression parser (Pratt-style precedence climbing) ---------------

    def _parse_expression(self, min_prec: int = 1):
        left = self._parse_prefix()
        while True:
            prec, op = self._get_binary_op()
            if prec is None or prec < min_prec:
                break
            if op in ("IS", "IN", "BETWEEN", "NOT_IN", "NOT_BETWEEN"):
                left = self._parse_postfix_op(left, op)
            else:
                self._advance()
                right = self._parse_expression(prec + 1)
                left = BinaryOp(op, left, right)
        return left

    def _get_binary_op(self) -> tuple:
        """Return (precedence, op_string) or (None, None)."""
        tok = self._peek()
        tt = tok.tt
        # Check NOT IN / NOT BETWEEN via two-token lookahead
        if tt == TT.NOT and self.pos + 1 < len(self.tokens):
            next_tt = self.tokens[self.pos + 1].tt
            if next_tt == TT.IN:
                return (4, "NOT_IN")
            if next_tt == TT.BETWEEN:
                return (4, "NOT_BETWEEN")
            return (None, None)

        mapping = {
            TT.OR: (1, "OR"), TT.AND: (2, "AND"),
            TT.EQ: (4, "="), TT.NE: (4, "<>"),
            TT.LT: (4, "<"), TT.GT: (4, ">"),
            TT.LE: (4, "<="), TT.GE: (4, ">="),
            TT.LIKE: (4, "LIKE"), TT.IS: (4, "IS"),
            TT.IN: (4, "IN"), TT.BETWEEN: (4, "BETWEEN"),
            TT.PLUS: (5, "+"), TT.MINUS: (5, "-"),
            TT.STAR: (6, "*"), TT.SLASH: (6, "/"),
        }
        return mapping.get(tt, (None, None))

    def _parse_prefix(self):
        if self._at(TT.NOT):
            self._advance()
            return UnaryOp("NOT", self._parse_expression(3))
        if self._at(TT.MINUS):
            self._advance()
            return UnaryOp("-", self._parse_expression(7))
        return self._parse_primary()

    def _parse_primary(self):
        tok = self._peek()

        # Parenthesized expression
        if tok.tt == TT.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(TT.RPAREN)
            return expr

        # Literals
        if tok.tt == TT.NUMBER:
            self._advance()
            return Literal(tok.value)
        if tok.tt == TT.STRING:
            self._advance()
            return Literal(tok.value)
        if tok.tt == TT.TRUE:
            self._advance()
            return Literal(True)
        if tok.tt == TT.FALSE:
            self._advance()
            return Literal(False)
        if tok.tt == TT.NULL:
            self._advance()
            return Literal(None)

        # Aggregate functions (COUNT, SUM, AVG, MIN, MAX)
        if tok.tt in _AGG_FUNCS:
            return self._parse_agg_func()

        # Identifier: column ref, table.col, or function call
        if tok.tt == TT.IDENT:
            self._advance()
            name = tok.value
            # Function call
            if self._at(TT.LPAREN):
                return self._parse_func_call(name)
            # table.column or table.*
            if self._match(TT.DOT):
                col_tok = self._peek()
                if col_tok.tt == TT.STAR:
                    self._advance()
                    return ColumnRef(name, "*")
                col_name = self._expect(TT.IDENT).value
                return ColumnRef(name, col_name)
            return ColumnRef(None, name)

        # Star (reachable in some expression contexts)
        if tok.tt == TT.STAR:
            self._advance()
            return ColumnRef(None, "*")

        raise ParseError(f"Unexpected {tok.tt.name} in expression", tok)

    def _parse_agg_func(self) -> FuncCall:
        tok = self._advance()
        name = tok.tt.name
        self._expect(TT.LPAREN)
        distinct = False
        args: list = []
        if self._at(TT.RPAREN):
            pass  # zero-arg (shouldn't normally happen for aggregates)
        elif tok.tt == TT.COUNT and self._at(TT.STAR):
            self._advance()
            args = [ColumnRef(None, "*")]
        else:
            if self._match(TT.DISTINCT):
                distinct = True
            args = self._parse_expression_list()
        self._expect(TT.RPAREN)
        return FuncCall(name, args, distinct)

    def _parse_func_call(self, name: str) -> FuncCall:
        self._expect(TT.LPAREN)
        args: list = []
        if not self._at(TT.RPAREN):
            args = self._parse_expression_list()
        self._expect(TT.RPAREN)
        return FuncCall(name.upper(), args)

    def _parse_postfix_op(self, left, op: str):
        if op == "IS":
            self._advance()  # consume IS
            negated = self._match(TT.NOT) is not None
            self._expect(TT.NULL)
            return IsNullExpr(left, negated)
        if op == "IN":
            self._advance()  # consume IN
            return self._parse_in_list(left, negated=False)
        if op == "BETWEEN":
            self._advance()  # consume BETWEEN
            return self._parse_between(left, negated=False)
        if op == "NOT_IN":
            self._advance()  # consume NOT
            self._advance()  # consume IN
            return self._parse_in_list(left, negated=True)
        if op == "NOT_BETWEEN":
            self._advance()  # consume NOT
            self._advance()  # consume BETWEEN
            return self._parse_between(left, negated=True)
        raise ParseError(f"Unknown postfix op {op}")  # pragma: no cover

    def _parse_in_list(self, left, negated: bool) -> InExpr:
        self._expect(TT.LPAREN)
        values = self._parse_expression_list()
        self._expect(TT.RPAREN)
        return InExpr(left, values, negated)

    def _parse_between(self, left, negated: bool) -> BetweenExpr:
        low = self._parse_expression(5)   # stop before AND at prec 2
        self._expect(TT.AND)
        high = self._parse_expression(5)
        return BetweenExpr(left, low, high, negated)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_sql(sql: str):
    """Tokenize and parse a SQL string, returning an AST node."""
    tokens = tokenize(sql)
    parser = Parser(tokens)
    return parser.parse()
