"""
Write-Ahead Log (WAL)
=====================
Append-only log file that records every page modification *before*
it reaches the data file, enabling crash recovery (redo/undo).
 
Log record format:
  [lsn:u64][txn_id:u32][type:u8][page_id:u32][len:u16][data:...][checksum:u32]
 
Record types:
  0 = BEGIN
  1 = UPDATE  (before-image / after-image pair)
  2 = COMMIT
  3 = ABORT
  4 = CHECKPOINT
  5 = CLR      (compensation log record for undo)
"""

from __future__ import annotations
 
import struct
import zlib
import threading
from enum import IntEnum
from pathlib import Path
 
from pydb import WAL_MAGIC

class LogRecordType(IntEnum):
    BEGIN      = 0
    UPDATE     = 1
    COMMIT     = 2
    ABORT      = 3
    CHECKPOINT = 4
    CLR        = 5
    
_REC_HDR = struct.Struct("<QIBIh")  # lsn(8)+txn_id(4)+type(1)+page_id(4)+data_len(2) = 19
 
class LogRecord:
    __slots__ = ("lsn", "txn_id", "rec_type", "page_id", "data", "checksum")
    
    def __init__(self, lsn: int, txn_id: int, rec_type: LogRecordType, page_id: int, data: bytes = b''):
        self.lsn = lsn
        self.txn_id = txn_id
        self.rec_type = rec_type
        self.page_id = page_id
        self.data = data
        self.checksum = 0
        
    def serialize(self) -> bytes:
        hdr = _REC_HDR.pack(self.lsn, self.txn_id, int(self.rec_type), self.page_id, len(self.data))
        payload = hdr + self.data
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        return payload + struct.pack("<I", crc)
    
    @classmethod
    def deserialize(cls, buf: bytes, offset: int = 0) -> tuple["LogRecord", int]:
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
    """Append-only write-ahead log backed by a single file."""
    
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = threading.Lock
        self._lsn = 0
        
        if self._path.exists() and self._path.stat().st_size > 0:
            self._fp = open(self._path, "r+b", buffering=0)
            self._recover_lsn()
        else:
            self._fp = open(self._path, "w+b", buffering=0)
            self._fp.write(WAL_MAGIC)
            
    def _recover_len(self):
        """Scan forward to find the highest LSN (simple crash-recovery)"""
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
        return self._lsn
    
    def flush(self):
        with self._lock:
            self._fp.flush()
    
    def iter_records(self):
        """Yield all valid LogRecords from the WAL (for recovery)."""
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
        """Clear the WAL (e.g. after a clean checkpoint)."""
        with self._lock:
            self._fp.seek(0)
            self._fp.truncate()
            self._fp.write(WAL_MAGIC)
            self._lsn = 0
 
    def close(self):
        self._fp.close()