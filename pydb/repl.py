"""
Interactive REPL
================
 
Overview
--------
The REPL (Read-Eval-Print Loop) is the primary user-facing interface
for interactive database sessions.  It reads SQL statements from
standard input, executes them against the ``Database`` engine, and
pretty-prints the results as ASCII tables.
 
Features
~~~~~~~~
* **readline support** — arrow-key navigation and command history
  (when the ``readline`` module is available).
* **Multi-line input** — SQL statements can span multiple lines.
  The REPL accumulates input until it sees a trailing ``;``, then
  executes the complete statement.
* **Dot-commands** — shell-like commands prefixed with ``.`` for
  metadata inspection and server control:
 
  ``.help``
      Show available commands.
  ``.tables``
      List all tables in the database.
  ``.schema TABLE``
      Show a table's column definitions and indexes.
  ``.indexes``
      List all indexes across all tables.
  ``.server [PORT]``
      Start the TCP server on *PORT* (default 5433) in a
      background thread, allowing remote clients to connect
      while the REPL remains interactive.
  ``.quit`` / ``.exit``
      Flush the database, stop the TCP server (if running),
      and exit.
 
Result formatting
~~~~~~~~~~~~~~~~~
Query results are displayed as bordered ASCII tables::
 
    +----+---------+-----+
    | id | name    | age |
    +----+---------+-----+
    | 1  | Alice   | 30  |
    | 2  | Bob     | 25  |
    +----+---------+-----+
    2 row(s)
 
Column widths auto-adjust to the widest value.  ``NULL`` values
are displayed as the string ``"NULL"``.
"""

from __future__ import annotations
 
import sys
import threading
from typing import Optional
 
from pydb.engine import Database
from pydb.server import TCPServer
 
try:
    import readline  # noqa: F401 — imported for side-effect (enables arrow keys)
except ImportError:
    readline = None

BANNER = r"""
  ╔═══════════════════════════════════════════╗
  ║   PyDB v1.0 — A Database from Scratch     ║
  ║   B+Tree · WAL · LRU-K · SQL · TCP        ║
  ╠═══════════════════════════════════════════╣
  ║  Type .help for commands, SQL to query.   ║
  ║  Press Ctrl+D or type .quit to exit.      ║
  ╚═══════════════════════════════════════════╝
"""

def _format_table(columns: list[str], rows: list[list]) -> str:
    """Render a result set as a bordered ASCII table.
 
    Parameters
    ----------
    columns : list[str]
        Column header names.
    rows : list[list]
        Row data.  Each inner list has one value per column.
 
    Returns
    -------
    str
        The formatted table, ready for ``print()``.
    """
    
    if not columns:
        return ""
    
    widths = [len(str(c)) for c in columns]
    str_rows = []
    for row in rows:
        sr = [str(v) if v is not None else "NULL" for v in row]
        str_rows.append(sr)
        
        for i, v in enumerate(sr):
            if i < len(widths):
                widths[i] = max(widths[i], len(v))
                
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(str(c).ljust(w) for c, w in zip(columns, widths)) + " |"
    lines = [sep, hdr, sep]
    for sr in str_rows:
        padded = []
        for i, v in enumerate(sr):
            w = widths[i] if i < len(widths) else len(v)
            padded.append(v.ljust(w))
        lines.append("| " + " | ".join(padded) + " |")
    lines.append(sep)
    return "\n".join(lines)

class REPL:
    """Interactive database REPL.
 
    Parameters
    ----------
    db_path : str
        Path to the database directory.  Created if it doesn't exist.
    """
 
    def __init__(self, db_path: str = "pydb_data"):
        self._db = Database(db_path)
        self._server: Optional[TCPServer] = None
        
    def run(self):
        """Enter the read-eval-print loop.  Blocks until exit."""
        print(BANNER)
        buf = ""
        while True:
            try:
                prompt = "pydb> " if not buf else "  ... "
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            
            stripped = line.strip()
            if not stripped:
                continue
            
            # Dot-commands are single-line execute immediately
            if not buf and stripped.startswith("."):
                self._dot_command(stripped)
                continue
            
            buf += (" " if buf else "") + line
 
            # Execute when the statement is complete (ends with ';')
            # or is a bare transaction keyword.
            if buf.rstrip().endswith(";"):
                self._execute(buf.rstrip())
                buf = ""
            elif stripped.upper() in ("BEGIN", "COMMIT", "ROLLBACK"):
                self._execute(buf.rstrip())
                buf = ""
                
    def _execute(self, sql: str):
        """Execute a SQL statement and print the result."""
        result = self._db.execute(sql)
        if result["rows"]:
            print(_format_table(result["columns"], result["rows"]))
        if result["message"]:
            print(result["message"])
 
    def _dot_command(self, cmd: str):
        """Dispatch a dot-command."""
        parts = cmd.split()
        name = parts[0].lower()
 
        if name in (".quit", ".exit"):
            self._shutdown()
            sys.exit(0)
 
        elif name == ".help":
            print("""
Special commands:
  .help              Show this help
  .tables            List all tables
  .schema <TABLE>    Show table schema and indexes
  .indexes           List all indexes
  .server [PORT]     Start TCP server (default port 5433)
  .quit / .exit      Exit
""")
 
        elif name == ".tables":
            tables = list(self._db._catalog.tables.values())
            if not tables:
                print("(no tables)")
            else:
                for t in tables:
                    print(f"  {t.name}")
 
        elif name == ".schema":
            if len(parts) < 2:
                print("Usage: .schema TABLE_NAME")
                return
            try:
                tdef = self._db._catalog.get_table(parts[1])
            except KeyError as e:
                print(f"Error: {e}")
                return
            print(f"Table: {tdef.name}  (heap page: {tdef.heap_page})")
            for c in tdef.columns:
                flags = []
                if c.primary_key:
                    flags.append("PRIMARY KEY")
                if not c.nullable:
                    flags.append("NOT NULL")
                fstr = " ".join(flags)
                print(f"  {c.name:20s} {c.col_type.name:10s} {fstr}")
            if tdef.indexes:
                print("Indexes:")
                for idx in tdef.indexes.values():
                    uq = "UNIQUE " if idx.unique else ""
                    print(f"  {idx.name}: {uq}({', '.join(idx.columns)})")
 
        elif name == ".indexes":
            indexes = list(self._db._catalog.indexes.values())
            if not indexes:
                print("(no indexes)")
            else:
                for ix in indexes:
                    uq = "UNIQUE " if ix.unique else ""
                    print(f"  {ix.name} ON {ix.table_name} {uq}({', '.join(ix.columns)})")
 
        elif name == ".server":
            port = int(parts[1]) if len(parts) > 1 else 5433
            if self._server:
                print("Server already running")
                return
            self._server = TCPServer(self._db, port=port,
                                     user_store=self._db.user_store)
            t = threading.Thread(target=self._server.start, daemon=True)
            t.start()
            print(f"TCP server started on port {port}")
 
        else:
            print(f"Unknown command: {name}  (try .help)")
 
    def _shutdown(self):
        """Stop the TCP server (if running) and close the database."""
        if self._server:
            self._server.stop()
        self._db.close()
        print("Database closed.")
        
def main():
    """CLI entry point — parses arguments and launches the REPL or server."""
    import argparse
    parser = argparse.ArgumentParser(description="PyDB — A Database from Scratch")
    parser.add_argument("--data", default="pydb_data", help="Database directory")
    parser.add_argument("--server", action="store_true", help="Start TCP server only (no REPL)")
    parser.add_argument("--port", type=int, default=5433, help="TCP server port")
    args = parser.parse_args()
 
    if args.server:
        db = Database(args.data)
        srv = TCPServer(db, port=args.port, user_store=db.user_store)
        print(f"Starting PyDB TCP server on port {args.port}...")
        try:
            srv.start()
        except KeyboardInterrupt:
            srv.stop()
            db.close()
    else:
        repl = REPL(args.data)
        repl.run()
 
 
if __name__ == "__main__":
    main()