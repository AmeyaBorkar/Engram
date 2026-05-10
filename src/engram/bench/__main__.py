"""`python -m engram.bench` entry point."""

from __future__ import annotations

import sys

from engram.bench._cli import main

if __name__ == "__main__":
    sys.exit(main())
