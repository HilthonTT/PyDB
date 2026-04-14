from pydb.page import SlottedPage, PageType
from pydb import PAGE_SIZE, HEADER_SIZE, INVALID_PAGE

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PyDB - Database from Scratch")
    parser.add_argument("--data", default='pydb_data', help='The database directory')
    parser.add_argument("--server", default='store_true', help='Start TCP server only (no REPL)')
    parser.add_argument('--port', default=5433, help='TCP server port')
    args = parser.parse_args()
    
    page = SlottedPage(page_id=42, page_type=PageType.DATA)

    # Read header back from the raw buffer into a fresh page
    page2 = SlottedPage(buf=bytearray(page._buf))

    assert page2.page_id     == 42
    assert page2.page_type   == PageType.DATA
    assert page2.num_slots   == 0
    assert page2.free_offset == HEADER_SIZE
    assert page2.free_end    == PAGE_SIZE
    assert page2.overflow_pid == INVALID_PAGE
    assert page2.lsn         == 0

    print("Header round-trip OK")
    print(f"  page_id={page2.page_id}, type={page2.page_type.name}, "
          f"free_offset={page2.free_offset}, free_end={page2.free_end}")
    
    print(args)
    
if __name__ == "__main__":
    main()
    