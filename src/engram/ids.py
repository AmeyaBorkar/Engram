"""Time-ordered identifiers (UUIDv7).

UUIDv7 (RFC 9562) carries a 48-bit millisecond timestamp prefix, so primary
keys are roughly time-ordered — which gives much better b-tree locality than
UUIDv4 on insert-heavy workloads. Python ships UUIDv7 generation in 3.14+;
we generate our own to support 3.10+.

Monotonicity inside a millisecond
---------------------------------
Two `new_id()` calls within the same millisecond would, with a purely random
`rand_a`, collide-or-invert ~1 in 4096 times. That breaks the "time-ordered"
invariant callers rely on for cursor pagination, b-tree insert locality,
and event-log ordering. RFC 9562 §6.2 method 1 ("monotonic random") fixes
this: within a given millisecond we treat `rand_a` as a 12-bit counter
seeded once and incremented on every call. On a millisecond boundary the
counter is re-seeded from a fresh random draw. Across a wall-clock regression
(NTP correction, process migration, ...) we treat the new millisecond as
authoritative but never emit a non-monotonic id -- if the wall clock moves
backwards we pin to the previous millisecond and continue counting.

The counter is protected by a Lock so concurrent callers see consistent
state. Generation cost is dominated by the lock + a single `os.urandom`
call on the millisecond boundary; in steady state inside one ms it is a
counter increment.
"""

from __future__ import annotations

import os
import threading
import time
from uuid import UUID

_VERSION_7 = 0x7
_VARIANT_RFC4122 = 0b10
_RAND_A_MAX = 0x0FFF  # 12 bits

# Mutable state for the §6.2 monotonic counter. Protected by `_LOCK`.
_LOCK = threading.Lock()
_LAST_MS = -1
_LAST_RAND_A = 0


def new_id() -> UUID:
    """Return a new UUIDv7 — time-ordered, millisecond precision.

    Per RFC 9562 §6.2 (method 1: monotonic random), `rand_a` acts as a
    counter inside a single millisecond so consecutive ids generated under
    1ms apart remain strictly increasing. The first id of a millisecond
    seeds the counter from `os.urandom(2)`; subsequent ids in the same
    millisecond increment it. A backward wall-clock step is masked: we
    advance into the *previous* millisecond's counter rather than emit a
    non-monotonic id.

    Layout (big-endian, 128 bits total):

        48 bits  unix_ts_ms
         4 bits  version (= 0x7)
        12 bits  rand_a (counter inside a millisecond, see above)
         2 bits  variant (= 0b10)
        62 bits  rand_b (random, fresh per call)
    """
    global _LAST_MS, _LAST_RAND_A

    now_ms = int(time.time() * 1000) & 0xFFFF_FFFF_FFFF

    with _LOCK:
        if now_ms > _LAST_MS:
            # Fresh millisecond: seed the counter from entropy. Mask to
            # ensure room to count up before saturation (we leave 1 bit
            # of headroom; collision after 2048 ids/ms is acceptable and
            # falls back to the saturation branch below).
            _LAST_MS = now_ms
            _LAST_RAND_A = int.from_bytes(os.urandom(2), "big") & 0x07FF
        else:
            # Same millisecond as the previous id, or the wall clock went
            # backwards. Either way we MUST advance to keep the id stream
            # monotonic. Pin to the recorded ms (which dominates a regressed
            # wall clock) and increment the counter.
            if _LAST_RAND_A < _RAND_A_MAX:
                _LAST_RAND_A += 1
            else:
                # Counter saturated inside a single ms (>= 4096 ids in
                # < 1ms is extreme but possible on a tight loop). Borrow
                # from the next ms so monotonicity holds; the next real
                # wall-clock ms will land at or past this borrowed value.
                _LAST_MS += 1
                _LAST_RAND_A = int.from_bytes(os.urandom(2), "big") & 0x07FF
            now_ms = _LAST_MS
        rand_a = _LAST_RAND_A

    rand_b = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)

    value = (
        (now_ms << 80) | (_VERSION_7 << 76) | (rand_a << 64) | (_VARIANT_RFC4122 << 62) | rand_b
    )
    return UUID(int=value)


def timestamp_ms(uid: UUID) -> int:
    """Extract the millisecond timestamp from a UUIDv7. Raises if not v7."""
    if uid.version != 7:
        raise ValueError(f"not a UUIDv7 (version={uid.version})")
    return uid.int >> 80
