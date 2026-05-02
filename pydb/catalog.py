"""
System Catalog & Record Encoding
=================================
 
Overview
--------
The catalog is the database's **data dictionary** — it stores the
definitions of every table and index in the system.  When the parser
resolves a table name, when the planner checks for usable indexes,
and when the executor encodes/decodes rows, they all consult the
catalog.
 
The catalog is held **in memory** and persisted as a JSON file
(``catalog.json``) inside the database directory.  On startup the
``Database`` engine loads it; on shutdown it writes it back.
 
Record encoding
~~~~~~~~~~~~~~~
Heap pages store rows as opaque byte blobs.  The ``encode_record``
/ ``decode_record`` functions convert between Python value lists and
the binary format:
 
.. list-table::
   :header-rows: 1
 
   * - Type
     - struct format
     - Bytes
     - Notes
   * - INTEGER
     - ``<q`` (signed 64-bit)
     - 8
     - Full ``int64`` range.
   * - FLOAT
     - ``<d`` (64-bit double)
     - 8
     - IEEE 754 double precision.
   * - BOOLEAN
     - ``<B`` (unsigned byte)
     - 1
     - ``1`` = True, ``0`` = False.
   * - TEXT
     - ``<H`` len + UTF-8 bytes
     - 2 + N
     - Up to 65 535 bytes of UTF-8.
 
A **null bitmap** precedes the column data — one bit per column,
packed into ``ceil(num_columns / 8)`` bytes.  A set bit means the
corresponding column value is ``NULL`` and no bytes are stored for
it in the column-data region.
 
Key encoding (sort-comparable bytes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
B+Tree keys must be comparable as raw bytes (lexicographic order)
and produce the same ordering as the typed values.  ``encode_key``
handles this with type-specific transforms:
 
* **Integers** — add ``2^63`` to shift the signed range into
  unsigned, then pack as big-endian ``>Q``.  Big-endian is essential:
  the most-significant byte comes first, so lexicographic comparison
  matches numeric comparison.
* **Floats** — apply the IEEE 754 sort transform: flip the sign bit
  for positives; complement all bits for negatives.  Pack big-endian.
* **Text** — escape any ``\\x00`` bytes in the UTF-8 string (since
  ``\\x00`` is the terminator), then append ``\\x00\\x00`` as the
  end-of-string marker.
* **Booleans** — ``\\x01`` for True, ``\\x00`` for False.
* **NULLs** — a single ``\\x00`` byte.  Non-null values are prefixed
  with ``\\x01``, so NULLs sort before all non-null values.
 
Supported SQL type aliases
~~~~~~~~~~~~~~~~~~~~~~~~~~
The parser accepts several aliases for each base type:
 
* ``INTEGER``, ``INT``
* ``FLOAT``, ``REAL``, ``DOUBLE``
* ``TEXT``, ``VARCHAR``, ``STRING``
* ``BOOLEAN``, ``BOOL``
"""

from __future__ import annotations
 
import json
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
 
from pydb import INVALID_PAGE
 

class ColType(Enum):
    """Enumeration of supported column data types.
 
    Members
    -------
    INTEGER : auto
        64-bit signed integer (``struct`` format ``q``).
    FLOAT : auto
        64-bit IEEE 754 double (``struct`` format ``d``).
    TEXT : auto
        Variable-length UTF-8 string (up to 65 535 bytes).
    BOOLEAN : auto
        Single byte — ``1`` for True, ``0`` for False.
    """
    INTEGER = auto()
    FLOAT = auto()
    TEXT = auto()
    BOOLEAN = auto()
    
_TYPE_MAP = {
    "INTEGER": ColType.INTEGER, "INT": ColType.INTEGER,
    "FLOAT": ColType.FLOAT, "REAL": ColType.FLOAT, "DOUBLE": ColType.FLOAT,
    "TEXT": ColType.TEXT, "VARCHAR": ColType.TEXT, "STRING": ColType.TEXT,
    "BOOLEAN": ColType.BOOLEAN, "BOOL": ColType.BOOLEAN,
}
"""Maps SQL type name strings (uppercased) to ``ColType`` values.
Supports common aliases like ``INT`` for ``INTEGER``, ``VARCHAR``
for ``TEXT``, etc."""
 
def parse_col_type(s: str) -> ColType:
    """Convert a SQL type string (e.g. ``"VARCHAR(255)"``) to a ``ColType``.
 
    The parenthesised length suffix (if present) is stripped — all
    text columns are variable-length with no enforced maximum.
 
    Raises
    ------
    ValueError
        If the type string is not recognised.
    """
    s = s.upper().split("(")[0].strip()
    ct = _TYPE_MAP.get(s)
    if ct is None:
        raise ValueError(f"Unknown column type: {s}")
    return ct

@dataclass
class Column:
    """Definition of a single table column.
 
    Attributes
    ----------
    name : str
        Column name as declared in ``CREATE TABLE``.
    col_type : ColType
        The column's data type.
    nullable : bool
        ``True`` if the column accepts ``NULL`` values.
    primary_key : bool
        ``True`` if this column is part of the primary key.
    """
    name: str
    col_type: ColType
    nullable: bool = True
    primary_key: bool = False
    
@dataclass
class IndexDef:
    """Definition of a B+Tree index.
 
    Attributes
    ----------
    name : str
        Index name (e.g. ``"pk_users"`` or ``"idx_age"``).
    table_name : str
        The table this index belongs to.
    columns : list[str]
        Column names that form the index key, in order.
    root_page : int
        Page id of the B+Tree root node.
    unique : bool
        ``True`` if the index enforces uniqueness.
    """
    name: str
    table_name: str
    columns: list[str]
    root_page: int = INVALID_PAGE
    unique: bool = False
    
@dataclass
class TableDef:
    """Definition of a table (schema + physical metadata).
 
    Attributes
    ----------
    name : str
        Table name as declared in ``CREATE TABLE``.
    columns : list[Column]
        Ordered list of column definitions.
    heap_page : int
        Page id of the first data page in this table's heap chain.
        ``INVALID_PAGE`` if the table has no data pages yet.
    next_rowid : int
        Auto-increment counter for integer primary keys.
    indexes : dict[str, IndexDef]
        Indexes defined on this table, keyed by index name.
    """
    name: str
    columns: list[Column]
    heap_page: int = INVALID_PAGE
    next_rowid: int = 1
    indexes: dict[str, "IndexDef"] = field(default_factory=dict)
    
    @property
    def col_names(self) -> list[str]:
        """Return column names in declaration order."""
        return [c.name for c in self.columns]
    
    def col_index(self, name: str) -> int:
        """Return the zero-based position of column *name* (case-insensitive).
 
        Raises
        ------
        KeyError
            If no column with that name exists in this table.
        """
        name_l = name.lower()
        for i, c in enumerate(self.columns):
            if c.name.lower() == name_l:
                return i
        raise KeyError(f"Column '{name}' not in table '{self.name}'")
    
class Catalog:
    """In-memory catalog of all table and index definitions.
 
    Table and index names are stored **lower-cased** as dictionary
    keys, making all lookups case-insensitive.
 
    Persistence is handled externally by the ``Database`` engine,
    which calls ``to_json()`` on shutdown and ``from_json()`` on
    startup.
    """
    
    def __init__(self):
        self.tables: dict[str, TableDef] = {}
        self.indexes: dict[str, IndexDef] = {}
        
    def create_table(self, tdef: TableDef):
        """Register a new table definition.
 
        Raises ``ValueError`` if a table with the same name already exists.
        """
        key = tdef.name.lower()
        if key in self.tables:
            raise ValueError(f"Table '{tdef.name}' already exists")
        self.tables[key] = tdef
        
    def get_table(self, name: str) -> TableDef:
        """Look up a table by name (case-insensitive).
 
        Raises ``KeyError`` if not found.
        """
        key = name.lower()
        t = self.tables.get(key)
        if t is None:
            raise KeyError(f"Table '{name}' does not exist")
        return t
    
    def drop_table(self, name):
        """Remove a table and all its associated indexes.
 
        Raises ``KeyError`` if the table does not exist.
        """
        key = name.lower()
        t = self.tables.pop(key, None)
        if t is None:
            raise KeyError(f"Table '{name}' does not exist")
        for idx_name in list(t.indexes):
            self.indexes.pop(idx_name.lower(), None)
        
    def create_index(self, idef: IndexDef):
        """Register a new index and attach it to its parent table.
 
        Raises ``ValueError`` if an index with the same name exists.
        """
        key = idef.name.lower()
        if key in self.indexes:
            raise ValueError(f"Index '{idef.name}' already exists")
        self.indexes[key] = idef
        tbl = self.get_table(idef.table_name)
        tbl.indexes[idef.name] = idef
        
    def get_index(self, name: str) -> IndexDef:
        """Look up an index by name (case-insensitive).
 
        Raises ``KeyError`` if not found.
        """
        key = name.lower()
        idx = self.indexes.get(key)
        if idx is None:
            raise KeyError(f"Index '{name}' does not exist")
        return idx
    
    def to_json(self) -> str:
        """Serialise the entire catalog to a JSON string.
 
        Called by ``Database.close()`` to persist the catalog to
        ``catalog.json``.  Index definitions are stored both at the
        top level and inside their parent table (the ``from_json``
        loader reconstructs both references).
        """
        d = {"tables": {}, "indexes": {}}
        for k, t in self.tables.items():
            d["tables"][k] = {
                "name": t.name,
                "columns": [
                    {"name": c.name, "type": c.col_type.name,
                     "nullable": c.nullable, "primary_key": c.primary_key}
                    for c in t.columns
                ],
                "heap_page": t.heap_page,
                "next_rowid": t.next_rowid,
            }
        for k, ix in self.indexes.items():
            d["indexes"][k] = {
                "name": ix.name,
                "table_name": ix.table_name,
                "columns": ix.columns,
                "root_page": ix.root_page,
                "unique": ix.unique,
            }
        return json.dumps(d)
    
    @classmethod
    def from_json(cls, s: str) -> "Catalog":
        """Reconstruct a ``Catalog`` from a JSON string.
 
        Called by the ``Database`` constructor when reopening an
        existing database directory.
        """
        d = json.loads(s)
        cat = cls()
        for k, td in d.get("tables", {}).items():
            cols = [Column(c["name"], ColType[c["type"]], c["nullable"], c["primary_key"])
                    for c in td["columns"]]
            tdef = TableDef(td["name"], cols, td["heap_page"], td["next_rowid"])
            cat.tables[k] = tdef
        for k, ix in d.get("indexes", {}).items():
            idef = IndexDef(ix["name"], ix["table_name"], ix["columns"],
                            ix["root_page"], ix["unique"])
            cat.indexes[k] = idef
            tbl = cat.tables.get(ix["table_name"].lower())
            if tbl:
                tbl.indexes[ix["name"]] = idef
        return cat
            
def encode_record(columns: list[Column], values: list[Any]) -> bytes:
    """Encode a row of Python values into a binary record for heap storage.
 
    The binary format is::
 
        [null_bitmap: ceil(N/8) bytes][col0_data][col1_data]...
 
    NULL values set their bit in the bitmap and consume zero bytes
    in the data region.  Non-null values are encoded according to
    their ``ColType``.
 
    Parameters
    ----------
    columns : list[Column]
        The table's column definitions (order matters).
    values : list[Any]
        Python values in the same order as *columns*.  ``None``
        represents SQL ``NULL``.
 
    Returns
    -------
    bytes
        The encoded record, ready to be inserted into a ``SlottedPage``.
    """
    n = len(columns)
    bm_bytes = (n + 7) // 8
    bitmap = bytearray(bm_bytes)
    parts: list[bytes] = []
    
    for i, (col, val) in enumerate(zip(columns, values)):
        if val is None:
            bitmap[i // 8] |= (1 << (i % 8))
            continue
        if col.col_type == ColType.INTEGER:
            parts.append(struct.pack("<q", int(val)))
        elif col.col_type == ColType.FLOAT:
            parts.append(struct.pack("<d", float(val)))
        elif col.col_type == ColType.BOOLEAN:
            parts.append(struct.pack("<B", 1 if val else 0))
        elif col.col_type == ColType.TEXT:
            raw = str(val).encode("utf-8")
            parts.append(struct.pack("<H", len(raw)) + raw)
 
    return bytes(bitmap) + b"".join(parts)

def decode_record(columns: list[Column], data: bytes) -> list[Any]:
    """Decode a binary record back into a list of Python values.
 
    The inverse of ``encode_record``.  NULL columns (indicated by
    the bitmap) are returned as ``None``.
 
    Parameters
    ----------
    columns : list[Column]
        The table's column definitions.
    data : bytes
        The raw record bytes from a ``SlottedPage``.
 
    Returns
    -------
    list[Any]
        Python values in column order.
    """
    n = len(columns)
    bm_bytes = (n + 7) // 8
    bitmap = data[:bm_bytes]
    pos = bm_bytes
    values: list[Any] = []
 
    for i, col in enumerate(columns):
        if bitmap[i // 8] & (1 << (i % 8)):
            values.append(None)
            continue
        if col.col_type == ColType.INTEGER:
            values.append(struct.unpack_from("<q", data, pos)[0]); pos += 8
        elif col.col_type == ColType.FLOAT:
            values.append(struct.unpack_from("<d", data, pos)[0]); pos += 8
        elif col.col_type == ColType.BOOLEAN:
            values.append(bool(data[pos])); pos += 1
        elif col.col_type == ColType.TEXT:
            slen = struct.unpack_from("<H", data, pos)[0]; pos += 2
            values.append(data[pos:pos + slen].decode("utf-8")); pos += slen
 
    return values

def encode_key(columns: list[Column], values: list[Any]) -> bytes:
    """Encode column values into a byte string that sorts lexicographically
    in the same order as the original typed values.
 
    This is used to produce B+Tree index keys.  The encoding must
    satisfy: for any two value tuples *A* and *B*,
    ``encode_key(A) < encode_key(B)`` **if and only if** *A* < *B*
    under the column types' natural ordering.
 
    The encoding scheme per type is documented in the module docstring.
 
    Parameters
    ----------
    columns : list[Column]
        Column definitions (determines the encoding per value).
    values : list[Any]
        The values to encode (same order as *columns*).
 
    Returns
    -------
    bytes
        A byte string suitable for lexicographic comparison.
        Uses **big-endian** packing (``>``) so that the most
        significant byte comes first.
    """
    parts: list[bytes] = []
    for col, val in zip(columns, values):
        if val is None:
            parts.append(b"\x00")
            continue
        parts.append(b"\x01")  # non-null marker
        if col.col_type == ColType.INTEGER:
            v = (int(val) + (1 << 63)) & 0xFFFFFFFFFFFFFFFF
            parts.append(struct.pack(">Q", v))
        elif col.col_type == ColType.FLOAT:
            f = float(val)
            bits = struct.unpack(">Q", struct.pack(">d", f))[0]
            if f >= 0:
                bits ^= (1 << 63)
            else:
                bits = ~bits & 0xFFFFFFFFFFFFFFFF
            parts.append(struct.pack(">Q", bits))
        elif col.col_type == ColType.BOOLEAN:
            parts.append(b"\x01" if val else b"\x00")
        elif col.col_type == ColType.TEXT:
            raw = str(val).encode("utf-8")
            escaped = raw.replace(b"\x00", b"\x00\x01")
            parts.append(escaped + b"\x00\x00")
    return b"".join(parts)