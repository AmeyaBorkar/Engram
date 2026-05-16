"""Time helpers shared across the codebase.

A handful of modules (`schemas`, `memory`, `decay/_engine`,
`reconcile/_engine`, `consolidation/_engine`) used to declare their
own `_utcnow()` — five identical copies that drift independently when
clock semantics evolve (e.g. switching to a monotonic-only timestamp
for replay tests).  Importing from here keeps them in sync.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Always uses `timezone.utc` so downstream tz-aware comparisons
    don't raise.  Don't use `datetime.utcnow()` — it returns a naive
    datetime and is deprecated in Python 3.12+.
    """
    return datetime.now(timezone.utc)
