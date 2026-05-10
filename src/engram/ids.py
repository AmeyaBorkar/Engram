"""Time-ordered identifiers (UUIDv7).

UUIDv7 (RFC 9562) carries a 48-bit millisecond timestamp prefix, so primary
keys are roughly time-ordered — which gives much better b-tree locality than
UUIDv4 on insert-heavy workloads. Python ships UUIDv7 generation in 3.14+;
we generate our own to support 3.10+.
"""

from __future__ import annotations

import os
import time
from uuid import UUID

_VERSION_7 = 0x7
_VARIANT_RFC4122 = 0b10


def new_id() -> UUID:
    """Return a new UUIDv7 — time-ordered, millisecond precision.

    Layout (big-endian, 128 bits total):

        48 bits  unix_ts_ms
         4 bits  version (= 0x7)
        12 bits  random (rand_a)
         2 bits  variant (= 0b10)
        62 bits  random (rand_b)
    """
    unix_ts_ms = int(time.time() * 1000) & 0xFFFF_FFFF_FFFF
    rand_a = int.from_bytes(os.urandom(2), "big") & 0x0FFF
    rand_b = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)

    value = (
        (unix_ts_ms << 80) | (_VERSION_7 << 76) | (rand_a << 64) | (_VARIANT_RFC4122 << 62) | rand_b
    )
    return UUID(int=value)


def timestamp_ms(uid: UUID) -> int:
    """Extract the millisecond timestamp from a UUIDv7. Raises if not v7."""
    if uid.version != 7:
        raise ValueError(f"not a UUIDv7 (version={uid.version})")
    return uid.int >> 80
