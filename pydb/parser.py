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
