"""Time-ordered identifiers (UUIDv7).

UUIDv7 (RFC 9562) carries a 48-bit millisecond timestamp prefix, so primary
keys are roughly time-ordered — which gives much better b-tree locality than
UUIDv4 on insert-heavy workloads. Python ships UUIDv7 generation in 3.14+;
we generate our own to support 3.10+.

Within a single millisecond we also enforce monotonicity via a 12-bit
`rand_a` counter (RFC 9562 §6.2 method 1): consecutive calls in the same ms
get strictly increasing rand_a values, and if the system clock steps
backwards we hold the previous timestamp. This means b-tree insertion
order under bursty workloads matches generation order.
"""

from __future__ import annotations

import os
import threading
import time
from uuid import UUID

_VERSION_7 = 0x7
_VARIANT_RFC4122 = 0b10
_RAND_A_MASK = 0x0FFF
_RAND_A_MAX = _RAND_A_MASK
_TS_MASK = 0xFFFF_FFFF_FFFF

# Monotonic state guarded by a lock.  Holding only a single tuple keeps the
# critical section small enough that contention is negligible for typical
# insert rates.
_STATE_LOCK = threading.Lock()
_LAST_TS_MS = 0
_LAST_RAND_A = 0


def _next_ts_and_counter() -> tuple[int, int]:
    """Return the (ts_ms, rand_a) tuple to use for the next id.

    Within a single millisecond `rand_a` increments monotonically.  If the
    counter would overflow we bump `ts_ms` by 1 (borrowing into the future)
    so b-tree order is still preserved.  If the wall clock steps backward
    we pin to the last observed timestamp.
    """
    global _LAST_TS_MS, _LAST_RAND_A
    now_ms = (time.time_ns() // 1_000_000) & _TS_MASK
    with _STATE_LOCK:
        if now_ms > _LAST_TS_MS:
            _LAST_TS_MS = now_ms
            _LAST_RAND_A = int.from_bytes(os.urandom(2), "big") & _RAND_A_MASK
        else:
            # Same or earlier millisecond — increment counter.  If we would
            # overflow the 12-bit field, advance the timestamp instead so
            # the resulting id still sorts after every previous one.
            if _LAST_RAND_A >= _RAND_A_MAX:
                _LAST_TS_MS = (_LAST_TS_MS + 1) & _TS_MASK
                _LAST_RAND_A = int.from_bytes(os.urandom(2), "big") & _RAND_A_MASK
            else:
                _LAST_RAND_A += 1
        return _LAST_TS_MS, _LAST_RAND_A


def new_id() -> UUID:
    """Return a new UUIDv7 — time-ordered, millisecond precision, monotonic.

    Layout (big-endian, 128 bits total):

        48 bits  unix_ts_ms
         4 bits  version (= 0x7)
        12 bits  rand_a (monotonic within a ms; RFC 9562 §6.2 method 1)
         2 bits  variant (= 0b10)
        62 bits  random (rand_b)
    """
    unix_ts_ms, rand_a = _next_ts_and_counter()
    rand_b = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)

    value = (
        (unix_ts_ms << 80) | (_VERSION_7 << 76) | (rand_a << 64) | (_VARIANT_RFC4122 << 62) | rand_b
    )
    return UUID(int=value)
