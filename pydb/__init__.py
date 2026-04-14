"""
PyDB — A relational database engine built from scratch in Python.
 
Components
----------
- Slotted-page storage engine with overflow pages
- LRU-K buffer-pool / cache replacement
- B+Tree indexes with split, merge, redistribute balancing
- WAL-based transaction manager (STEAL / NO-FORCE)
- Recursive-descent SQL parser producing a typed AST
- Cost-based query planner with index selection
- Volcano-style pull executor
- K-way external merge sort
- Length-prefixed TCP wire protocol
- Interactive REPL
"""

__version__ = "1.0.0"

# Global constants
PAGE_SIZE = 4096                    # bytes
INVALID_PAGE = 0xFFFFFFFF
HEADER_SIZE = 24                    # page header bytes
SLOT_ENTRY_SIZE = 4                 # (offset: u16, length: u16)
MAX_RECORD_SIZE = PAGE_SIZE - HEADER_SIZE - SLOT_ENTRY_SIZE
WAL_MAGIC       = b"PWAL"
DATA_MAGIC      = b"PYDB"
BTREE_ORDER     = 128               # max keys per B+Tree node
LRU_K           = 2                 # history depth for LRU-K
BUFFER_POOL_CAP = 1024              # default # of frames
SORT_MEM_PAGES  = 64                # pages of RAM for external sort runs
WIRE_MAX_MSG    = 16 * 1024 * 1024  # 16 MiB max network message
