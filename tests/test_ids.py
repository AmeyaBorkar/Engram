"""Tests for `engram.ids`."""

from __future__ import annotations

import time

from engram.ids import new_id


def test_new_id_returns_uuidv7() -> None:
    uid = new_id()
    assert uid.version == 7
    assert (uid.int >> 62) & 0b11 == 0b10  # RFC 4122 variant


def test_new_id_unique() -> None:
    ids = {new_id() for _ in range(10_000)}
    assert len(ids) == 10_000


def test_new_id_time_ordered() -> None:
    a = new_id()
    time.sleep(0.002)
    b = new_id()
    assert a.bytes < b.bytes, "UUIDv7 should be time-ordered when generation gap > 1ms"
