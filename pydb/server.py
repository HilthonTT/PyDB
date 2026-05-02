"""
TCP Server & Wire Protocol
===========================
 
Overview
--------
The TCP server allows remote clients to connect to the database and
execute SQL statements over a network socket.  This transforms PyDB
from a single-process embedded database into a client/server system
that supports multiple concurrent connections.
 
Wire protocol
~~~~~~~~~~~~~
The protocol is intentionally simple — a **length-prefixed** message
format with little-endian byte order::
 
    Client → Server:
      ┌──────────────────┬────────────────────────┐
      │ payload_len (u32)│ SQL string (UTF-8)     │
      │     4 bytes      │ payload_len bytes      │
      └──────────────────┴────────────────────────┘
 
    Server → Client:
      ┌──────────────────┬────────────────────────┐
      │ payload_len (u32)│ JSON result (UTF-8)    │
      │     4 bytes      │ payload_len bytes      │
      └──────────────────┴────────────────────────┘
 
The JSON result object has the shape::
 
    {
        "columns": ["id", "name", ...],
        "rows":    [[1, "Alice"], [2, "Bob"]],
        "message": "2 row(s)"
    }
 
Special commands:
 
* ``".quit"`` — closes the client connection gracefully.
 
The maximum message size is ``WIRE_MAX_MSG`` (16 MiB).  Messages
exceeding this are rejected to prevent memory exhaustion.
 
Concurrency model
~~~~~~~~~~~~~~~~~
Each accepted TCP connection spawns a **daemon thread** running a
``ClientHandler``.  All handlers share the same ``Database`` instance,
which is internally synchronised by the buffer pool's lock and the
transaction manager's lock manager.
 
The server socket has a 1-second accept timeout so that ``stop()``
can be called from another thread and takes effect within ~1 second.
 
Client helper
~~~~~~~~~~~~~
The ``PyDBClient`` class provides a convenient Python-native client
for the wire protocol, useful for testing and scripting::
 
    client = PyDBClient("127.0.0.1", 5433)
    result = client.execute("SELECT * FROM users;")
    print(result["rows"])
    client.close()
"""

from __future__ import annotations
 
import json
import socket
import struct
import threading
import time
from typing import Optional
 
from pydb import WIRE_MAX_MSG
from pydb.engine import Database

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from *sock*, blocking until complete.
 
    Raises ``ConnectionError`` if the peer disconnects before all
    bytes are received.
    """
    buf = bytearray()
    while len(buf) < n:
        chunck = sock.recv(n - len(buf))
        if not chunck:
            raise ConnectionError("Client disconnected")
        buf.extend(chunck)
    return bytes(buf)

def _read_message(sock: socket.socket) -> str:
    """Read one length-prefixed message and return the UTF-8 payload.
 
    Raises ``ValueError`` if the declared length exceeds ``WIRE_MAX_MSG``.
    """
    hdr = _recv_exact(sock, 4)
    length = struct.unpack("<I", hdr)[0]
    if length > WIRE_MAX_MSG:
        raise ValueError(f"Message too large: {length} bytes")
    payload = _recv_exact(sock, length)
    return payload.decode("utf-8")

def _send_message(sock: socket.socket, data: str):
    """Send a length-prefixed UTF-8 message.
 
    The 4-byte length header is written in a single ``sendall`` call
    together with the payload to avoid Nagle-related latency.
    """
    payload = data.encode("utf-8")
    sock.sendall(struct.pack("<I", len(payload)) + payload)
    
class ClientHandler(threading.Thread):
    """Handles one client connection in a dedicated daemon thread.
 
    Reads SQL statements in a loop, executes them against the shared
    ``Database``, and sends back JSON results.  The loop terminates
    on disconnect or when the client sends ``".quit"``.
    """
    
    def __init__(self, sock: socket.socket, addr, db: Database):
        super().__init__(daemon=True)
        self._sock = sock
        self._addr = addr
        self._db = db
        
    def run(self):
        print(f"[server] Client connected: {self._addr}")
        try:
            while True:
                try:
                    sql = _read_message(self._sock)
                except ConnectionError:
                    break
                    
                sql = sql.strip()
                if not sql:
                    continue
                if sql.lower() == ".quit":
                    _send_message(self._sock, json.dumps(
                        {"columns": [], "rows": [], "message": "Goodbye"}))
                    break
            
                result = self._db.execute(sql)
                safe_rows = []
                for row in result.get("rows", []):
                    safe_rows.append([
                        v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
                        for v in row
                    ])
                result["rows"] = safe_rows
                _send_message(self._sock, json.dumps(result))
        except Exception as e:
            print(f"[server] Error with {self._addr}: {e}")
        finally:
            self._sock.close()
            print(f"[server] Client disconnected: {self._addr}")
            
class TCPServer:
    """Multi-threaded TCP server for database connections.
 
    Parameters
    ----------
    db : Database
        The database engine to execute queries against.
    host : str
        Bind address.  ``"0.0.0.0"`` listens on all interfaces.
    port : int
        TCP port number.  Default ``5433`` (one above PostgreSQL's
        default, to avoid conflicts).
    """
    
    def __init__(self, db: Database, host: str = "0.0.0.0", port: int = 5433):
        self._db = db
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._running = False
        
    def start(self):
        """Start accepting connections.  Blocks until ``stop()`` is called."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._sock.listen(32)
        self._sock.settimeout(1.0)
        self._running = True
        print(f"[server] Listening on {self._host}:{self._port}")
        
        while self._running:
            try:
                client_sock, addr = self._sock.accept()
                handler = ClientHandler(client_sock, addr, self._db)
                handler.start()
            except socket.timeout:
                continue
            except OSError:
                break
            
    def stop(self):
        """Signal the server to stop accepting new connections.
 
        Existing client threads will finish their current request
        and then terminate when the client disconnects.
        """
        self._running = False
        if self._sock:
            self._sock.close()
            
class PyDBClient:
    """Blocking TCP client for the PyDB wire protocol.
 
    Connects to a running ``TCPServer`` and provides a simple
    ``execute(sql) -> dict`` interface.
 
    Parameters
    ----------
    host : str
        Server hostname or IP address.
    port : int
        Server port number.
    """
 
    def __init__(self, host: str = "127.0.0.1", port: int = 5433):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))
 
    def execute(self, sql: str) -> dict:
        """Send a SQL statement and return the JSON result.
 
        Returns
        -------
        dict
            ``{"columns": [...], "rows": [...], "message": "..."}``.
        """
        _send_message(self._sock, sql)
        resp = _read_message(self._sock)
        return json.loads(resp)
 
    def close(self):
        """Send ``".quit"`` and close the socket."""
        try:
            _send_message(self._sock, ".quit")
            _read_message(self._sock)
        except Exception:
            pass
        self._sock.close()
