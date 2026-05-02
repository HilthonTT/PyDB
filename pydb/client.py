"""
PyDB Client Library
===================

Overview
--------
A standalone TCP client for PyDB's wire protocol, designed for use
by application code (web servers, scripts, etc.).  Provides both a
simple single-connection ``Client`` and a thread-safe
``ConnectionPool`` for concurrent workloads.

Single connection
~~~~~~~~~~~~~~~~~
::

    from pydb.client import Client

    with Client("127.0.0.1", 5433) as db:
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        db.execute("INSERT INTO users (name) VALUES ('Alice')")
        result = db.execute("SELECT * FROM users")
        for row in result["rows"]:
            print(row)

Connection pool
~~~~~~~~~~~~~~~
::

    from pydb.client import ConnectionPool

    pool = ConnectionPool("127.0.0.1", 5433, min_size=2, max_size=10)

    # Each `acquire` returns a Client; `release` puts it back.
    with pool.connection() as db:
        result = db.execute("SELECT COUNT(*) FROM users")

    pool.close()

Wire protocol
~~~~~~~~~~~~~
Messages are length-prefixed, little-endian::

    [payload_len: u32][payload: UTF-8 bytes]

Client sends SQL strings; server replies with JSON::

    {"columns": [...], "rows": [...], "message": "..."}
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
from contextlib import contextmanager
from typing import Optional

from pydb import WIRE_MAX_MSG


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from *sock*."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Server disconnected")
        buf.extend(chunk)
    return bytes(buf)


def _read_message(sock: socket.socket) -> str:
    """Read one length-prefixed message and return the UTF-8 payload."""
    hdr = _recv_exact(sock, 4)
    length = struct.unpack("<I", hdr)[0]
    if length > WIRE_MAX_MSG:
        raise ValueError(f"Message too large: {length} bytes")
    payload = _recv_exact(sock, length)
    return payload.decode("utf-8")


def _send_message(sock: socket.socket, data: str):
    """Send a length-prefixed UTF-8 message."""
    payload = data.encode("utf-8")
    sock.sendall(struct.pack("<I", len(payload)) + payload)


class Client:
    """Single-connection PyDB client with context-manager support.

    Parameters
    ----------
    host : str
        Server hostname or IP address.
    port : int
        Server port number.
    timeout : float or None
        Socket timeout in seconds.  ``None`` means blocking.

    Examples
    --------
    ::

        with Client("127.0.0.1", 5433) as db:
            result = db.execute("SELECT 1 + 1")
            print(result["rows"])  # [[2]]
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5433,
                 timeout: Optional[float] = None):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._connect()

    def _connect(self):
        """Establish the TCP connection."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self._timeout is not None:
            self._sock.settimeout(self._timeout)
        self._sock.connect((self._host, self._port))

    def execute(self, sql: str) -> dict:
        """Send a SQL statement and return the parsed JSON result.

        Parameters
        ----------
        sql : str
            A single SQL statement.

        Returns
        -------
        dict
            ``{"columns": list[str], "rows": list[list], "message": str}``

        Raises
        ------
        ConnectionError
            If the server disconnected.
        """
        if self._sock is None:
            raise ConnectionError("Client is closed")
        _send_message(self._sock, sql)
        resp = _read_message(self._sock)
        return json.loads(resp)

    @property
    def connected(self) -> bool:
        """Whether the underlying socket is open."""
        return self._sock is not None

    def close(self):
        """Send ``.quit`` and close the socket."""
        if self._sock is None:
            return
        try:
            _send_message(self._sock, ".quit")
            _read_message(self._sock)
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
        self._sock = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self):
        state = "connected" if self._sock else "closed"
        return f"Client({self._host}:{self._port}, {state})"


class ConnectionPool:
    """Thread-safe pool of ``Client`` connections.

    Maintains a queue of idle connections.  When a connection is
    requested, the pool returns an idle one or creates a new one
    (up to ``max_size``).  Connections are returned to the pool
    after use.

    Parameters
    ----------
    host : str
        Server hostname or IP address.
    port : int
        Server port number.
    min_size : int
        Number of connections to open eagerly on construction.
    max_size : int
        Maximum total connections.  If all are in use and a new
        one is requested, the caller blocks until one is returned.
    timeout : float or None
        Socket timeout for each connection.

    Examples
    --------
    ::

        pool = ConnectionPool("127.0.0.1", 5433, min_size=2, max_size=10)

        with pool.connection() as db:
            rows = db.execute("SELECT * FROM users")["rows"]

        pool.close()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5433,
                 min_size: int = 2, max_size: int = 10,
                 timeout: Optional[float] = None):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._max_size = max(min_size, max_size)
        self._pool: queue.Queue[Client] = queue.Queue(maxsize=self._max_size)
        self._size = 0
        self._lock = threading.Lock()
        self._closed = False

        # Pre-populate with min_size connections
        for _ in range(min_size):
            self._pool.put(self._new_conn())

    def _new_conn(self) -> Client:
        """Create a new connection and track pool size."""
        with self._lock:
            self._size += 1
        return Client(self._host, self._port, self._timeout)

    def acquire(self) -> Client:
        """Get a connection from the pool (blocking if needed).

        Returns
        -------
        Client
            A connected client.  The caller **must** call
            ``release`` when done, or use the ``connection()``
            context manager.

        Raises
        ------
        RuntimeError
            If the pool has been closed.
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        # Try to get an idle connection
        try:
            conn = self._pool.get_nowait()
            if conn.connected:
                return conn
            # Dead connection — discard and create a new one
            with self._lock:
                self._size -= 1
        except queue.Empty:
            pass

        # Can we create a new one?
        with self._lock:
            if self._size < self._max_size:
                return self._new_conn()

        # Pool exhausted — block until one is returned
        conn = self._pool.get(timeout=30)
        if not conn.connected:
            with self._lock:
                self._size -= 1
            return self._new_conn()
        return conn

    def release(self, conn: Client):
        """Return a connection to the pool.

        If the pool is full or closed, the connection is closed
        instead.
        """
        if self._closed or not conn.connected:
            conn.close()
            with self._lock:
                self._size -= 1
            return
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            conn.close()
            with self._lock:
                self._size -= 1

    @contextmanager
    def connection(self):
        """Context manager that acquires and releases a connection.

        Yields
        ------
        Client
            A connected client.

        Examples
        --------
        ::

            with pool.connection() as db:
                db.execute("INSERT INTO logs (msg) VALUES ('hello')")
        """
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)

    def close(self):
        """Close all idle connections and mark the pool as closed.

        Connections currently in use are closed when they are returned.
        """
        self._closed = True
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
                with self._lock:
                    self._size -= 1
            except queue.Empty:
                break

    def __repr__(self):
        idle = self._pool.qsize()
        return f"ConnectionPool({self._host}:{self._port}, size={self._size}, idle={idle})"
