"""Tests for the SQL parser (tokenizer + recursive-descent parser)."""

import pytest

from pydb.parser import (
    TT, Token, ParseError, tokenize, parse_sql,
    ColumnRef, Literal, BinaryOp, UnaryOp, FuncCall,
    IsNullExpr, InExpr, BetweenExpr,
    SelectColumn, OrderByItem, JoinClause, ColumnDef,
    SelectStmt, InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt, CreateIndexStmt,
    BeginStmt, CommitStmt, RollbackStmt, ExplainStmt,
)


# =====================================================================
# Tokenizer
# =====================================================================

class TestTokenize:
    def test_keywords(self):
        tokens = tokenize("SELECT FROM WHERE")
        assert tokens[0].tt == TT.SELECT
        assert tokens[1].tt == TT.FROM
        assert tokens[2].tt == TT.WHERE
        assert tokens[3].tt == TT.EOF

    def test_identifiers(self):
        tokens = tokenize("users age_col _private")
        assert all(t.tt == TT.IDENT for t in tokens[:3])
        assert tokens[0].value == "users"
        assert tokens[1].value == "age_col"
        assert tokens[2].value == "_private"

    def test_keyword_case_insensitive(self):
        tokens = tokenize("select SELECT Select")
        assert all(t.tt == TT.SELECT for t in tokens[:3])

    def test_integer(self):
        tokens = tokenize("42")
        assert tokens[0].tt == TT.NUMBER
        assert tokens[0].value == 42
        assert isinstance(tokens[0].value, int)

    def test_float(self):
        tokens = tokenize("3.14")
        assert tokens[0].tt == TT.NUMBER
        assert tokens[0].value == 3.14
        assert isinstance(tokens[0].value, float)

    def test_string(self):
        tokens = tokenize("'hello world'")
        assert tokens[0].tt == TT.STRING
        assert tokens[0].value == "hello world"

    def test_string_escaped_quote(self):
        tokens = tokenize("'O''Brien'")
        assert tokens[0].tt == TT.STRING
        assert tokens[0].value == "O'Brien"

    def test_multi_char_operators(self):
        tokens = tokenize("<> <= >=")
        assert tokens[0].tt == TT.NE
        assert tokens[1].tt == TT.LE
        assert tokens[2].tt == TT.GE

    def test_single_char_symbols(self):
        tokens = tokenize("( ) , . ; * = < > + - /")
        expected = [TT.LPAREN, TT.RPAREN, TT.COMMA, TT.DOT, TT.SEMI,
                    TT.STAR, TT.EQ, TT.LT, TT.GT, TT.PLUS, TT.MINUS, TT.SLASH]
        for tok, exp in zip(tokens, expected):
            assert tok.tt == exp

    def test_whitespace_skipped(self):
        tokens = tokenize("  SELECT  *  ")
        assert tokens[0].tt == TT.SELECT
        assert tokens[1].tt == TT.STAR
        assert tokens[2].tt == TT.EOF

    def test_eof_sentinel(self):
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].tt == TT.EOF

    def test_position_tracking(self):
        tokens = tokenize("SELECT *")
        assert tokens[0].pos == 0
        assert tokens[1].pos == 7

    def test_unexpected_character(self):
        with pytest.raises(ParseError, match="Unexpected character"):
            tokenize("SELECT @")


# =====================================================================
# Statements
# =====================================================================

class TestCreateTable:
    def test_basic(self):
        r = parse_sql("CREATE TABLE users (id INT, name TEXT)")
        assert isinstance(r, CreateTableStmt)
        assert r.name == "users"
        assert len(r.columns) == 2
        assert r.columns[0].name == "id"
        assert r.columns[0].type_name == "INT"
        assert r.columns[1].name == "name"
        assert r.columns[1].type_name == "TEXT"

    def test_primary_key(self):
        r = parse_sql("CREATE TABLE t (id INT PRIMARY KEY)")
        assert r.columns[0].primary_key is True
        assert r.columns[0].nullable is False

    def test_not_null(self):
        r = parse_sql("CREATE TABLE t (name TEXT NOT NULL)")
        assert r.columns[0].nullable is False
        assert r.columns[0].primary_key is False

    def test_varchar_with_length(self):
        r = parse_sql("CREATE TABLE t (name VARCHAR(255) NOT NULL)")
        assert r.columns[0].type_name == "VARCHAR(255)"
        assert r.columns[0].nullable is False

    def test_multiple_constraints(self):
        r = parse_sql("CREATE TABLE t (id INT PRIMARY KEY NOT NULL)")
        assert r.columns[0].primary_key is True
        assert r.columns[0].nullable is False


class TestCreateIndex:
    def test_basic(self):
        r = parse_sql("CREATE INDEX idx_name ON users (name)")
        assert isinstance(r, CreateIndexStmt)
        assert r.name == "idx_name"
        assert r.table == "users"
        assert r.columns == ["name"]
        assert r.unique is False

    def test_unique(self):
        r = parse_sql("CREATE UNIQUE INDEX idx_email ON users (email)")
        assert r.unique is True

    def test_multi_column(self):
        r = parse_sql("CREATE INDEX idx_comp ON t (a, b, c)")
        assert r.columns == ["a", "b", "c"]


class TestDropTable:
    def test_basic(self):
        r = parse_sql("DROP TABLE users")
        assert isinstance(r, DropTableStmt)
        assert r.name == "users"


class TestInsert:
    def test_single_row(self):
        r = parse_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        assert isinstance(r, InsertStmt)
        assert r.table == "users"
        assert r.columns == ["id", "name"]
        assert len(r.values) == 1
        assert r.values[0][0] == Literal(1)
        assert r.values[0][1] == Literal("Alice")

    def test_multi_row(self):
        r = parse_sql("INSERT INTO t (a, b) VALUES (1, 2), (3, 4)")
        assert len(r.values) == 2
        assert r.values[0] == [Literal(1), Literal(2)]
        assert r.values[1] == [Literal(3), Literal(4)]


class TestSelect:
    def test_star(self):
        r = parse_sql("SELECT * FROM t")
        assert isinstance(r, SelectStmt)
        assert len(r.columns) == 1
        assert r.columns[0].expr == ColumnRef(None, "*")
        assert r.from_table == "t"

    def test_columns(self):
        r = parse_sql("SELECT a, b FROM t")
        assert len(r.columns) == 2
        assert r.columns[0].expr == ColumnRef(None, "a")
        assert r.columns[1].expr == ColumnRef(None, "b")

    def test_alias_with_as(self):
        r = parse_sql("SELECT name AS n FROM t")
        assert r.columns[0].alias == "n"

    def test_alias_implicit(self):
        r = parse_sql("SELECT name n FROM t")
        assert r.columns[0].alias == "n"

    def test_table_alias(self):
        r = parse_sql("SELECT * FROM users u")
        assert r.from_alias == "u"

    def test_distinct(self):
        r = parse_sql("SELECT DISTINCT name FROM t")
        assert r.distinct is True

    def test_where(self):
        r = parse_sql("SELECT * FROM t WHERE x > 5")
        assert isinstance(r.where, BinaryOp)
        assert r.where.op == ">"

    def test_group_by(self):
        r = parse_sql("SELECT dept, COUNT(*) FROM t GROUP BY dept")
        assert len(r.group_by) == 1
        assert r.group_by[0] == ColumnRef(None, "dept")

    def test_having(self):
        r = parse_sql("SELECT dept FROM t GROUP BY dept HAVING COUNT(*) > 1")
        assert isinstance(r.having, BinaryOp)

    def test_order_by_asc(self):
        r = parse_sql("SELECT * FROM t ORDER BY name ASC")
        assert len(r.order_by) == 1
        assert r.order_by[0].descending is False

    def test_order_by_desc(self):
        r = parse_sql("SELECT * FROM t ORDER BY name DESC")
        assert r.order_by[0].descending is True

    def test_order_by_default_asc(self):
        r = parse_sql("SELECT * FROM t ORDER BY name")
        assert r.order_by[0].descending is False

    def test_order_by_multiple(self):
        r = parse_sql("SELECT * FROM t ORDER BY a DESC, b ASC")
        assert len(r.order_by) == 2
        assert r.order_by[0].descending is True
        assert r.order_by[1].descending is False

    def test_limit(self):
        r = parse_sql("SELECT * FROM t LIMIT 10")
        assert r.limit == Literal(10)

    def test_limit_offset(self):
        r = parse_sql("SELECT * FROM t LIMIT 10 OFFSET 5")
        assert r.limit == Literal(10)
        assert r.offset == Literal(5)

    def test_complex(self):
        sql = ("SELECT DISTINCT u.name, COUNT(*) AS cnt "
               "FROM users u LEFT JOIN orders o ON u.id = o.user_id "
               "WHERE age > 18 GROUP BY u.name HAVING cnt > 1 "
               "ORDER BY cnt DESC LIMIT 10 OFFSET 5")
        r = parse_sql(sql)
        assert r.distinct is True
        assert r.from_table == "users"
        assert r.from_alias == "u"
        assert len(r.joins) == 1
        assert r.joins[0].join_type == "LEFT"
        assert r.joins[0].table == "orders"
        assert r.joins[0].alias == "o"
        assert isinstance(r.where, BinaryOp)
        assert len(r.group_by) == 1
        assert isinstance(r.having, BinaryOp)
        assert len(r.order_by) == 1
        assert r.order_by[0].descending is True
        assert r.limit == Literal(10)
        assert r.offset == Literal(5)


class TestJoins:
    def test_inner_join(self):
        r = parse_sql("SELECT * FROM a INNER JOIN b ON a.id = b.id")
        assert r.joins[0].join_type == "INNER"

    def test_bare_join(self):
        r = parse_sql("SELECT * FROM a JOIN b ON a.id = b.id")
        assert r.joins[0].join_type == "INNER"

    def test_left_join(self):
        r = parse_sql("SELECT * FROM a LEFT JOIN b ON a.id = b.id")
        assert r.joins[0].join_type == "LEFT"

    def test_right_join(self):
        r = parse_sql("SELECT * FROM a RIGHT JOIN b ON a.id = b.id")
        assert r.joins[0].join_type == "RIGHT"

    def test_cross_join(self):
        r = parse_sql("SELECT * FROM a CROSS JOIN b")
        assert r.joins[0].join_type == "CROSS"
        assert r.joins[0].condition is None

    def test_join_with_alias(self):
        r = parse_sql("SELECT * FROM a JOIN b x ON a.id = x.id")
        assert r.joins[0].alias == "x"

    def test_multiple_joins(self):
        r = parse_sql("SELECT * FROM a JOIN b ON a.id = b.id LEFT JOIN c ON b.id = c.id")
        assert len(r.joins) == 2
        assert r.joins[0].join_type == "INNER"
        assert r.joins[1].join_type == "LEFT"


class TestUpdate:
    def test_basic(self):
        r = parse_sql("UPDATE users SET name = 'Bob' WHERE id = 1")
        assert isinstance(r, UpdateStmt)
        assert r.table == "users"
        assert len(r.assignments) == 1
        assert r.assignments[0] == ("name", Literal("Bob"))
        assert isinstance(r.where, BinaryOp)

    def test_multiple_assignments(self):
        r = parse_sql("UPDATE t SET a = 1, b = 2")
        assert len(r.assignments) == 2
        assert r.where is None


class TestDelete:
    def test_with_where(self):
        r = parse_sql("DELETE FROM users WHERE age < 18")
        assert isinstance(r, DeleteStmt)
        assert r.table == "users"
        assert isinstance(r.where, BinaryOp)

    def test_without_where(self):
        r = parse_sql("DELETE FROM users")
        assert r.where is None


class TestTransactionControl:
    def test_begin(self):
        assert isinstance(parse_sql("BEGIN"), BeginStmt)

    def test_commit(self):
        assert isinstance(parse_sql("COMMIT"), CommitStmt)

    def test_rollback(self):
        assert isinstance(parse_sql("ROLLBACK"), RollbackStmt)


class TestExplain:
    def test_explain_select(self):
        r = parse_sql("EXPLAIN SELECT * FROM users")
        assert isinstance(r, ExplainStmt)
        assert isinstance(r.stmt, SelectStmt)
        assert r.stmt.from_table == "users"


class TestTrailingSemicolon:
    def test_accepted(self):
        r = parse_sql("SELECT * FROM t;")
        assert isinstance(r, SelectStmt)

    def test_without_semicolon(self):
        r = parse_sql("SELECT * FROM t")
        assert isinstance(r, SelectStmt)


# =====================================================================
# Expressions
# =====================================================================

class TestArithmetic:
    def test_precedence_add_mul(self):
        """1 + 2 * 3 should parse as +(1, *(2, 3))."""
        r = parse_sql("SELECT 1 + 2 * 3 FROM t")
        expr = r.columns[0].expr
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"
        assert expr.left == Literal(1)
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.op == "*"

    def test_precedence_mul_add(self):
        """2 * 3 + 1 should parse as +(*(2, 3), 1)."""
        r = parse_sql("SELECT 2 * 3 + 1 FROM t")
        expr = r.columns[0].expr
        assert expr.op == "+"
        assert isinstance(expr.left, BinaryOp)
        assert expr.left.op == "*"
        assert expr.right == Literal(1)

    def test_parenthesized(self):
        """(1 + 2) * 3 should parse as *(+(1, 2), 3)."""
        r = parse_sql("SELECT (1 + 2) * 3 FROM t")
        expr = r.columns[0].expr
        assert expr.op == "*"
        assert isinstance(expr.left, BinaryOp)
        assert expr.left.op == "+"

    def test_division(self):
        r = parse_sql("SELECT a / b FROM t")
        expr = r.columns[0].expr
        assert expr.op == "/"

    def test_subtraction(self):
        r = parse_sql("SELECT a - b FROM t")
        expr = r.columns[0].expr
        assert expr.op == "-"


class TestComparison:
    @pytest.mark.parametrize("op_sql,op_str", [
        ("=", "="), ("<>", "<>"), ("<", "<"), (">", ">"), ("<=", "<="), (">=", ">="),
    ])
    def test_operators(self, op_sql, op_str):
        r = parse_sql(f"SELECT * FROM t WHERE a {op_sql} b")
        assert isinstance(r.where, BinaryOp)
        assert r.where.op == op_str


class TestBoolean:
    def test_and(self):
        r = parse_sql("SELECT * FROM t WHERE a = 1 AND b = 2")
        assert r.where.op == "AND"
        assert r.where.left.op == "="
        assert r.where.right.op == "="

    def test_or(self):
        r = parse_sql("SELECT * FROM t WHERE a = 1 OR b = 2")
        assert r.where.op == "OR"

    def test_not(self):
        r = parse_sql("SELECT * FROM t WHERE NOT x = 1")
        assert isinstance(r.where, UnaryOp)
        assert r.where.op == "NOT"

    def test_precedence_and_or(self):
        """a OR b AND c should parse as OR(a, AND(b, c))."""
        r = parse_sql("SELECT * FROM t WHERE a = 1 OR b = 2 AND c = 3")
        assert r.where.op == "OR"
        assert r.where.right.op == "AND"

    def test_precedence_not_and(self):
        """NOT a AND b should parse as AND(NOT(a), b)."""
        r = parse_sql("SELECT * FROM t WHERE NOT a = 1 AND b = 2")
        assert r.where.op == "AND"
        assert isinstance(r.where.left, UnaryOp)
        assert r.where.left.op == "NOT"


class TestUnaryMinus:
    def test_negative_literal(self):
        r = parse_sql("SELECT * FROM t WHERE x = -1")
        assert isinstance(r.where.right, UnaryOp)
        assert r.where.right.op == "-"
        assert r.where.right.operand == Literal(1)

    def test_negative_in_expression(self):
        r = parse_sql("SELECT -a + b FROM t")
        expr = r.columns[0].expr
        assert expr.op == "+"
        assert isinstance(expr.left, UnaryOp)


class TestLike:
    def test_like(self):
        r = parse_sql("SELECT * FROM t WHERE name LIKE 'A%'")
        assert isinstance(r.where, BinaryOp)
        assert r.where.op == "LIKE"
        assert r.where.right == Literal("A%")


class TestIsNull:
    def test_is_null(self):
        r = parse_sql("SELECT * FROM t WHERE x IS NULL")
        assert isinstance(r.where, IsNullExpr)
        assert r.where.negated is False

    def test_is_not_null(self):
        r = parse_sql("SELECT * FROM t WHERE x IS NOT NULL")
        assert isinstance(r.where, IsNullExpr)
        assert r.where.negated is True


class TestIn:
    def test_in(self):
        r = parse_sql("SELECT * FROM t WHERE x IN (1, 2, 3)")
        assert isinstance(r.where, InExpr)
        assert r.where.negated is False
        assert r.where.values == [Literal(1), Literal(2), Literal(3)]

    def test_not_in(self):
        r = parse_sql("SELECT * FROM t WHERE x NOT IN (1, 2)")
        assert isinstance(r.where, InExpr)
        assert r.where.negated is True


class TestBetween:
    def test_between(self):
        r = parse_sql("SELECT * FROM t WHERE x BETWEEN 1 AND 10")
        assert isinstance(r.where, BetweenExpr)
        assert r.where.negated is False
        assert r.where.low == Literal(1)
        assert r.where.high == Literal(10)

    def test_not_between(self):
        r = parse_sql("SELECT * FROM t WHERE x NOT BETWEEN 1 AND 10")
        assert isinstance(r.where, BetweenExpr)
        assert r.where.negated is True

    def test_between_does_not_consume_boolean_and(self):
        """BETWEEN 1 AND 10 AND y > 0 — the second AND is boolean."""
        r = parse_sql("SELECT * FROM t WHERE x BETWEEN 1 AND 10 AND y > 0")
        assert isinstance(r.where, BinaryOp)
        assert r.where.op == "AND"
        assert isinstance(r.where.left, BetweenExpr)
        assert isinstance(r.where.right, BinaryOp)
        assert r.where.right.op == ">"


class TestColumnRef:
    def test_qualified(self):
        r = parse_sql("SELECT t.col FROM t")
        assert r.columns[0].expr == ColumnRef("t", "col")

    def test_table_star(self):
        r = parse_sql("SELECT t.* FROM t")
        assert r.columns[0].expr == ColumnRef("t", "*")

    def test_unqualified(self):
        r = parse_sql("SELECT col FROM t")
        assert r.columns[0].expr == ColumnRef(None, "col")


class TestFunctions:
    def test_user_function(self):
        r = parse_sql("SELECT upper(name) FROM t")
        expr = r.columns[0].expr
        assert isinstance(expr, FuncCall)
        assert expr.name == "UPPER"
        assert len(expr.args) == 1

    def test_count_star(self):
        r = parse_sql("SELECT COUNT(*) FROM t")
        expr = r.columns[0].expr
        assert isinstance(expr, FuncCall)
        assert expr.name == "COUNT"
        assert expr.args == [ColumnRef(None, "*")]
        assert expr.distinct is False

    def test_count_distinct(self):
        r = parse_sql("SELECT COUNT(DISTINCT name) FROM t")
        expr = r.columns[0].expr
        assert expr.name == "COUNT"
        assert expr.distinct is True
        assert expr.args == [ColumnRef(None, "name")]

    def test_sum(self):
        r = parse_sql("SELECT SUM(amount) FROM t")
        expr = r.columns[0].expr
        assert expr.name == "SUM"

    def test_avg(self):
        r = parse_sql("SELECT AVG(price) FROM t")
        expr = r.columns[0].expr
        assert expr.name == "AVG"

    def test_min_max(self):
        r = parse_sql("SELECT MIN(a), MAX(b) FROM t")
        assert r.columns[0].expr.name == "MIN"
        assert r.columns[1].expr.name == "MAX"


# =====================================================================
# Literals
# =====================================================================

class TestLiterals:
    def test_true(self):
        r = parse_sql("SELECT * FROM t WHERE x = TRUE")
        assert r.where.right == Literal(True)

    def test_false(self):
        r = parse_sql("SELECT * FROM t WHERE x = FALSE")
        assert r.where.right == Literal(False)

    def test_null(self):
        r = parse_sql("SELECT * FROM t WHERE x = NULL")
        assert r.where.right == Literal(None)

    def test_string_literal(self):
        r = parse_sql("SELECT * FROM t WHERE x = 'hello'")
        assert r.where.right == Literal("hello")

    def test_integer_literal(self):
        r = parse_sql("SELECT * FROM t WHERE x = 42")
        assert r.where.right == Literal(42)

    def test_float_literal(self):
        r = parse_sql("SELECT * FROM t WHERE x = 3.14")
        assert r.where.right == Literal(3.14)


# =====================================================================
# Error handling
# =====================================================================

class TestErrors:
    def test_unexpected_token(self):
        with pytest.raises(ParseError):
            parse_sql("SELECT FROM")  # missing column list or *

    def test_unexpected_character(self):
        with pytest.raises(ParseError, match="Unexpected character"):
            parse_sql("SELECT @invalid FROM t")

    def test_error_has_position(self):
        with pytest.raises(ParseError, match="position"):
            parse_sql("SELECT")  # missing FROM

    def test_missing_rparen(self):
        with pytest.raises(ParseError):
            parse_sql("SELECT * FROM t WHERE x IN (1, 2")

    def test_garbage_after_statement(self):
        with pytest.raises(ParseError, match="end of input"):
            parse_sql("SELECT * FROM t; DROP TABLE t")

    def test_empty_input(self):
        with pytest.raises(ParseError):
            parse_sql("")

    def test_unique_create_table(self):
        with pytest.raises(ParseError, match="UNIQUE"):
            parse_sql("CREATE UNIQUE TABLE t (id INT)")
