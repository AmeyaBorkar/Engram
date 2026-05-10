"""Tests for `engram.ids`."""

from __future__ import annotations

import time

import pytest

from engram.ids import new_id, timestamp_ms


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


def test_timestamp_ms_recent() -> None:
    before = int(time.time() * 1000)
    uid = new_id()
    after = int(time.time() * 1000)
    ts = timestamp_ms(uid)
    assert before <= ts <= after


def test_timestamp_ms_rejects_non_v7() -> None:
    import uuid

    with pytest.raises(ValueError, match="not a UUIDv7"):
        timestamp_ms(uuid.uuid4())
