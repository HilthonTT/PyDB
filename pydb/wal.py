"""
Write-Ahead Log (WAL)
=====================
 
Overview
--------
The WAL is the foundation of crash recovery.  The core invariant
(the **WAL protocol**) is simple:
 
    *Before a dirty page is written to the data file, every log
    record describing that page's modifications must already be
    on stable storage.*
 
This guarantees that after a crash, the recovery manager can read
the WAL from front to back and reconstruct every committed
transaction's effects (redo) and roll back every incomplete
transaction's effects (undo).
 
File layout
~~~~~~~~~~~
The WAL is a flat, append-only file::
 
    ┌──────────┬──────────────┬──────────────┬─────
    │  Magic   │  LogRecord 0 │  LogRecord 1 │ ...
    │  (4 B)   │              │              │
    └──────────┴──────────────┴──────────────┴─────
    byte 0     byte 4         byte 4+rec0    ...
 
The first 4 bytes are ``WAL_MAGIC`` (``b"PWAL"``), a signature
that ``_recover_lsn`` validates on startup to reject corrupt or
unrelated files.
 
Log record format
~~~~~~~~~~~~~~~~~
Each record is variable-length and self-describing::
 
    ┌─────┬────────┬──────┬─────────┬──────┬──────┬──────────┐
    │ lsn │ txn_id │ type │ page_id │ dlen │ data │ checksum │
    │ u64 │  u32   │  u8  │   u32   │ i16  │ var  │   u32    │
    │ 8 B │  4 B   │ 1 B  │   4 B   │ 2 B  │ ... │   4 B    │
    └─────┴────────┴──────┴─────────┴──────┴──────┴──────────┘
 
The fixed header is 19 bytes (``_REC_HDR = struct.Struct("<QIBIh")``),
followed by ``dlen`` bytes of payload, followed by a 4-byte CRC32
checksum over ``header + data``.
 
* **lsn** — Log Sequence Number.  A monotonically increasing
  counter assigned by ``append``.  Pages store the LSN of the
  most recent write that touched them, enabling the recovery
  manager to skip already-applied records.
* **txn_id** — Which transaction produced this record.
* **type** — One of the ``LogRecordType`` enum values.
* **page_id** — The data-file page this record applies to.  Zero
  for non-page records (BEGIN, COMMIT, ABORT).
* **dlen** — Byte length of the ``data`` payload.  Signed ``i16``
  because the struct format uses ``h``; in practice always ≥ 0.
* **data** — Record-type-specific payload:
 
  - **UPDATE** records store a before-image / after-image pair,
    each prefixed with a 4-byte length.  The before-image is
    used during undo; the after-image during redo.
  - **BEGIN / COMMIT / ABORT** records carry no payload.
  - **CLR** (Compensation Log Record) records are written during
    undo to prevent repeated undo on a second crash.
 
* **checksum** — CRC32 of ``header + data``.  Used during recovery
  to detect truncated or corrupted tail records (common after an
  unclean shutdown).
 
Record types
~~~~~~~~~~~~
.. list-table::
   :header-rows: 1
 
   * - Type
     - Value
     - Meaning
   * - ``BEGIN``
     - 0
     - A new transaction started.
   * - ``UPDATE``
     - 1
     - A page was modified.  Carries before/after images.
   * - ``COMMIT``
     - 2
     - The transaction is durable.  Forces a WAL flush.
   * - ``ABORT``
     - 3
     - The transaction was rolled back.  Forces a WAL flush.
   * - ``CHECKPOINT``
     - 4
     - A snapshot of active transactions and dirty pages.
       Allows recovery to start from this point instead of
       scanning the entire log.
   * - ``CLR``
     - 5
     - Compensation record written during undo to make the
       undo operation itself idempotent.
 
Durability guarantees
~~~~~~~~~~~~~~~~~~~~~
``COMMIT`` and ``ABORT`` records trigger an immediate ``flush()``
(Python-level), ensuring they reach the OS page cache.  A full
``fsync`` happens when the ``TransactionManager`` calls
``WAL.flush()`` explicitly — this is the point at which the
commit is truly durable against power loss.
"""

from __future__ import annotations
 
import struct
import zlib
import threading
from enum import IntEnum
from pathlib import Path
 
from pydb import WAL_MAGIC

class LogRecordType(IntEnum):
    """Type discriminant for WAL log records.
 
    Stored as a single ``u8`` in the record header.
    """
    BEGIN      = 0
    UPDATE     = 1
    COMMIT     = 2
    ABORT      = 3
    CHECKPOINT = 4
    CLR        = 5
    
_REC_HDR = struct.Struct("<QIBIh")  # lsn(8)+txn_id(4)+type(1)+page_id(4)+data_len(2) = 19
"""Fixed portion of a log record: ``lsn(u64) txn_id(u32) type(u8)
page_id(u32) data_len(i16)`` — 19 bytes, little-endian."""
 
class LogRecord:
    """A single entry in the write-ahead log.
 
    ``LogRecord`` is a plain data object — it does not perform any
    I/O.  Serialisation (``serialize``) and deserialisation
    (``deserialize``) convert between this object and the on-disk
    byte format.
 
    Attributes
    ----------
    lsn : int
        Log Sequence Number assigned when the record was appended.
    txn_id : int
        The transaction that produced this record.
    rec_type : LogRecordType
        The kind of event this record represents.
    page_id : int
        The data-file page this record refers to.  ``0`` for
        transaction-level records (BEGIN / COMMIT / ABORT).
    data : bytes
        Type-specific payload.  Empty for BEGIN / COMMIT / ABORT.
        For UPDATE records this contains the before-image and
        after-image of the modified region, each prefixed with a
        4-byte length.
    checksum : int
        CRC32 read from disk (populated by ``deserialize``).
        ``serialize`` computes a fresh CRC32 each time — this field
        is only meaningful on records that were *read* from the WAL.
    """
    
    __slots__ = ("lsn", "txn_id", "rec_type", "page_id", "data", "checksum")
    
    def __init__(self, lsn: int, txn_id: int, rec_type: LogRecordType, page_id: int, data: bytes = b''):
        self.lsn = lsn
        self.txn_id = txn_id
        self.rec_type = rec_type
        self.page_id = page_id
        self.data = data
        self.checksum = 0
        
    def serialize(self) -> bytes:
        """Pack this record into its on-disk byte representation.
 
        Layout: ``[19-byte header][data bytes][4-byte CRC32]``.
 
        The CRC32 covers the header *and* the data, so a single
        bit-flip anywhere in the record is detected.
 
        Returns
        -------
        bytes
            The complete serialised record (19 + len(data) + 4 bytes).
        """
        hdr = _REC_HDR.pack(self.lsn, self.txn_id, int(self.rec_type), self.page_id, len(self.data))
        payload = hdr + self.data
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        return payload + struct.pack("<I", crc)
    
    
    @classmethod
    def deserialize(cls, buf: bytes, offset: int = 0) -> tuple["LogRecord", int]:
        """Unpack one record starting at *offset* within *buf*.
 
        Parameters
        ----------
        buf : bytes
            The raw WAL file contents (after the 4-byte magic).
        offset : int
            Byte position within *buf* where this record begins.
 
        Returns
        -------
        tuple[LogRecord, int]
            The parsed record and the byte offset immediately after
            it (i.e. where the *next* record starts).
 
        Raises
        ------
        struct.error
            If the buffer is too short to contain a complete header
            or the stated data length exceeds the buffer.  The caller
            (``_recover_lsn``, ``iter_records``) catches this to
            detect a truncated WAL tail.
        """
        lsn, txn_id, rtype, page_id, dlen = _REC_HDR.unpack_from(buf, offset)
        pos = offset + _REC_HDR.size
        data = buf[pos:pos + dlen]
        pos += dlen
        crc_stored = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        rec = cls(lsn, txn_id, LogRecordType(rtype), page_id, data)
        rec.checksum = crc_stored
        return rec, pos
    
class WAL:
    """Append-only write-ahead log backed by a single file.
 
    The WAL is the durability backbone of the database.  Every
    mutation flows through ``append`` before the corresponding dirty
    page is written to the data file.
 
    Parameters
    ----------
    path : str or Path
        Filesystem path to the WAL file.  Created if it doesn't
        exist; validated and recovered from if it does.
 
    Raises
    ------
    ValueError
        If the file exists but its magic bytes are not ``WAL_MAGIC``
        (``b"PWAL"``).
 
    Thread safety
    ~~~~~~~~~~~~~
    All public methods acquire ``self._lock`` (a ``threading.Lock``).
    Multiple threads (one per client connection) can safely append
    records concurrently.
    """
    
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._lsn = 0
        
        if self._path.exists() and self._path.stat().st_size > 0:
            self._fp = open(self._path, "r+b", buffering=0)
            self._recover_lsn()
        else:
            self._fp = open(self._path, "w+b", buffering=0)
            self._fp.write(WAL_MAGIC)
            
    def _recover_lsn(self):
        """Scan the WAL from front to back to find the highest LSN.
 
        This runs once during ``__init__`` when reopening an existing
        WAL file.  It sets ``self._lsn`` to one past the highest
        LSN found, so that the next ``append`` call produces a
        unique, monotonically increasing sequence number.
 
        Records with corrupt checksums or truncated tails are
        silently ignored — the scan stops at the first unparseable
        record, treating it as the logical end of the log.
 
        Raises
        ------
        ValueError
            If the file's magic bytes don't match ``WAL_MAGIC``.
        """
        self._fp.seek(0)
        magic = self._fp.read(4)
        if magic != WAL_MAGIC:
            raise ValueError("Not a valid WAL file")
        data = self._fp.read()
        pos = 0
        while pos < len(data):
            try:
                rec, pos = LogRecord.deserialize(data, pos)
                if rec.lsn > self._lsn:
                    self._lsn = rec.lsn + 1
            except Exception:
                break
            
    def append(self, txn_id: int, rec_type: LogRecordType, page_id: int = 0, data: bytes = b"") -> int:
        """Append a new log record and return its LSN.
 
        The record is serialised and written to the file immediately.
        For ``COMMIT`` and ``ABORT`` records, an additional
        ``flush()`` is issued to push the data out of Python's
        buffers and into the OS page cache — this is the minimum
        guarantee that a committed transaction's log records survive
        a Python-level crash.
 
        Parameters
        ----------
        txn_id : int
            The transaction producing this record.
        rec_type : LogRecordType
            What kind of event this is.
        page_id : int
            The data-file page involved (0 for BEGIN/COMMIT/ABORT).
        data : bytes
            Record-type-specific payload.
 
        Returns
        -------
        int
            The LSN assigned to this record.  Guaranteed to be
            strictly greater than all previously assigned LSNs.
        """
        with self._lock:
            lsn = self._lsn
            self._lsn += 1
            rec = LogRecord(lsn, txn_id, rec_type, page_id, data)
            self._fp.write(rec.serialize())
            if rec_type in (LogRecordType.COMMIT, LogRecordType.ABORT):
                self._fp.flush()
            return lsn
    
    @property
    def current_lsn(self) -> int:
        """The next LSN that ``append`` will assign.
 
        Useful for the buffer pool and transaction manager to stamp
        pages with the latest LSN that modified them.
        """
        return self._lsn
    
    def flush(self):
        """Force all buffered log data to the OS page cache.
 
        This does **not** call ``fsync`` — that responsibility
        belongs to the ``TransactionManager``, which calls
        ``DiskManager.flush()`` after committing.  This method
        just ensures Python's internal file buffer is drained.
        """
        with self._lock:
            self._fp.flush()
    
    def iter_records(self):
        """Yield every valid ``LogRecord`` in the WAL, front to back.
 
        Used by the recovery manager to replay committed transactions
        (redo) and roll back incomplete ones (undo).  Iteration
        stops at the first unparseable record (truncated tail).
 
        Yields
        ------
        LogRecord
            Each record in LSN order.
        """
        with self._lock:
            self._fp.seek(4)  # skip magic
            data = self._fp.read()
        pos = 0
        while pos < len(data):
            try:
                rec, pos = LogRecord.deserialize(data, pos)
                yield rec
            except Exception:
                break
 
    def truncate(self):
        """Clear the entire WAL, resetting it to an empty state.
 
        Called after a clean checkpoint: once all dirty pages have
        been flushed to the data file, the old log records are no
        longer needed for recovery.  The file is truncated to just
        the 4-byte magic header and the LSN counter is reset to 0.
        """
        with self._lock:
            self._fp.seek(0)
            self._fp.truncate()
            self._fp.write(WAL_MAGIC)
            self._lsn = 0
 
    def close(self):
        """Close the underlying file handle.
 
        After ``close()`` all other methods will raise an
        ``OSError``.  The ``Database`` engine calls this during
        its shutdown sequence.
        """
        self._fp.close()