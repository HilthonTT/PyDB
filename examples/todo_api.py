"""
Todo Web API — PyDB example
===========================

A minimal REST API for a todo-list application, backed by a running PyDB
TCP server. Uses only the Python standard library (``http.server``) so
the example matches PyDB's no-third-party-dependency philosophy.

Setup
-----
1. Start a PyDB server in one terminal::

       python -m pydb.main --data todo_db --server --port 5433

   On first launch a default ``admin``/``admin`` user is created.

2. In another terminal, run this example::

       python examples/todo_api.py

   The API listens on http://127.0.0.1:8000.

Endpoints
---------
* ``GET    /todos``           — list all todos
* ``POST   /todos``           — create todo  (JSON: {"title", "description"})
* ``GET    /todos/<id>``      — fetch one todo
* ``PUT    /todos/<id>``      — update todo  (JSON: any of title/description/completed)
* ``DELETE /todos/<id>``      — delete todo

Example
-------
::

    curl -X POST http://127.0.0.1:8000/todos \\
         -H 'Content-Type: application/json' \\
         -d '{"title": "Buy milk", "description": "2L whole"}'

    curl http://127.0.0.1:8000/todos
"""

from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Allow running this example directly with `python examples/todo_api.py`
# from the project root, without needing to install pydb as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydb.client import ConnectionPool  # noqa: E402  (sys.path tweak above)


DB_HOST = "127.0.0.1"
DB_PORT = 5433
DB_USER = "admin"
DB_PASS = "admin"

API_HOST = "127.0.0.1"
API_PORT = 8000

def sql_str(value: str) -> str:
    """Quote a Python string for safe inclusion in SQL.

    PyDB follows the SQL standard: a single quote inside a string literal
    is escaped by doubling it (``''``). Since the client library does not
    support parameterised queries, callers must escape user input here
    before interpolating into a SQL statement.
    """
    return "'" + value.replace("'", "''") + "'"


def sql_bool(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def row_to_todo(columns: list[str], row: list) -> dict:
    """Convert a (columns, row) pair from a result set into a todo dict."""
    return dict(zip(columns, row))

def init_schema(pool: ConnectionPool) -> None:
    """Create the todos table on first run.

    PyDB does not yet have ``CREATE TABLE IF NOT EXISTS``, so we simply
    catch the "already exists" error returned by the server.
    """
    create_sql = (
        "CREATE TABLE todos ("
        "  id INTEGER PRIMARY KEY,"
        "  title TEXT NOT NULL,"
        "  description TEXT,"
        "  completed BOOLEAN NOT NULL"
        ")"
    )
    with pool.connection() as db:
        result = db.execute(create_sql)
        msg: str = result.get("message", "")
        if "already exists" in msg.lower():
            return
        # Index for faster lookup-by-id (PRIMARY KEY already gives us one,
        # but illustrate explicit index creation for the example).

class TodoStore:
    """Thin SQL wrapper around the todos table."""

    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def list_all(self) -> list[dict]:
        with self._pool.connection() as db:
            res = db.execute("SELECT id, title, description, completed FROM todos ORDER BY id")
        return [row_to_todo(res["columns"], r) for r in res["rows"]]

    def get(self, todo_id: int) -> dict | None:
        sql = f"SELECT id, title, description, completed FROM todos WHERE id = {int(todo_id)}"
        with self._pool.connection() as db:
            res = db.execute(sql)
        if not res["rows"]:
            return None
        return row_to_todo(res["columns"], res["rows"][0])

    def create(self, title: str, description: str = "") -> dict:
        # Wrap in a transaction so the SELECT after INSERT sees the row we
        # just wrote and no other writer can sneak ahead in between.
        insert_sql = (
            "INSERT INTO todos (title, description, completed) VALUES "
            f"({sql_str(title)}, {sql_str(description)}, FALSE)"
        )
        with self._pool.connection() as db:
            db.execute("BEGIN")
            try:
                db.execute(insert_sql)
                res = db.execute(
                    "SELECT id, title, description, completed FROM todos "
                    "ORDER BY id DESC LIMIT 1"
                )
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        return row_to_todo(res["columns"], res["rows"][0])

    def update(self, todo_id: int, fields: dict) -> dict | None:
        assignments = []
        if "title" in fields and fields["title"] is not None:
            # title is NOT NULL — silently drop a null rather than crashing.
            assignments.append(f"title = {sql_str(fields['title'])}")
        if "description" in fields:
            v = fields["description"]
            assignments.append(
                "description = NULL" if v is None else f"description = {sql_str(v)}"
            )
        if "completed" in fields and fields["completed"] is not None:
            assignments.append(f"completed = {sql_bool(bool(fields['completed']))}")
        if not assignments:
            return self.get(todo_id)

        sql = f"UPDATE todos SET {', '.join(assignments)} WHERE id = {int(todo_id)}"
        with self._pool.connection() as db:
            db.execute(sql)
        return self.get(todo_id)

    def delete(self, todo_id: int) -> bool:
        with self._pool.connection() as db:
            res = db.execute(f"DELETE FROM todos WHERE id = {int(todo_id)}")
        # The server reports e.g. "1 row(s) deleted"; parse out the count.
        msg = res.get("message", "0")
        m = re.match(r"(\d+)", msg)
        return bool(m and int(m.group(1)) > 0)


class TodoHandler(BaseHTTPRequestHandler):
    store: TodoStore  # injected via class attribute in main()

    def _json(self, status: int, body) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _match_id(self) -> int | None:
        m = re.match(r"^/todos/(\d+)/?$", self.path)
        return int(m.group(1)) if m else None

    def do_GET(self):  # noqa: N802 — http.server API
        if self.path.rstrip("/") == "/todos":
            self._json(200, self.store.list_all())
            return
        todo_id = self._match_id()
        if todo_id is not None:
            todo = self.store.get(todo_id)
            if todo is None:
                self._json(404, {"error": "not found"})
            else:
                self._json(200, todo)
            return
        self._json(404, {"error": "no such route"})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/todos":
            self._json(404, {"error": "no such route"})
            return
        body = self._read_json()
        title = (body.get("title") or "").strip()
        if not title:
            self._json(400, {"error": "title is required"})
            return
        description = body.get("description") or ""
        todo = self.store.create(title, description)
        self._json(201, todo)

    def do_PUT(self):  # noqa: N802
        todo_id = self._match_id()
        if todo_id is None:
            self._json(404, {"error": "no such route"})
            return
        body = self._read_json()
        # Allow only known fields through.
        fields = {k: body[k] for k in ("title", "description", "completed") if k in body}
        todo = self.store.update(todo_id, fields)
        if todo is None:
            self._json(404, {"error": "not found"})
        else:
            self._json(200, todo)

    def do_DELETE(self):  # noqa: N802
        todo_id = self._match_id()
        if todo_id is None:
            self._json(404, {"error": "no such route"})
            return
        if self.store.delete(todo_id):
            self._json(204, {})
        else:
            self._json(404, {"error": "not found"})

    def log_message(self, fmt, *args):  # silence default access log
        return


def main() -> None:
    pool = ConnectionPool(
        DB_HOST, DB_PORT,
        min_size=2, max_size=10,
        username=DB_USER, password=DB_PASS,
    )
    init_schema(pool)
    TodoHandler.store = TodoStore(pool)

    server = ThreadingHTTPServer((API_HOST, API_PORT), TodoHandler)
    print(f"Todo API listening on http://{API_HOST}:{API_PORT}")
    print(f"Backed by PyDB at {DB_HOST}:{DB_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        server.server_close()
        pool.close()


if __name__ == "__main__":
    main()
