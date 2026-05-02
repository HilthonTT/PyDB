"""PyDB package entry point.
 
Allows the database to be launched with ``python -m pydb``.
Delegates immediately to ``repl.main()`` which parses CLI arguments
and starts either the interactive REPL or the TCP-only server.
"""
from pydb.repl import main

main()
 