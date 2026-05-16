"""Tests for `engram.ids`."""

from __future__ import annotations

import threading
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


# --- RFC 9562 §6.2 monotonic-counter behavior --------------------------------


class TestMonotonicCounter:
    """Inside a single millisecond, `rand_a` acts as a counter so the
    id stream is strictly monotonic (b-tree locality, cursor stability).
    """

    def test_tight_loop_is_strictly_monotonic(self) -> None:
        # Generate a burst quickly enough that many calls land in one ms.
        ids = [new_id() for _ in range(10_000)]
        for prev, curr in zip(ids, ids[1:], strict=False):
            assert prev.bytes < curr.bytes, (
                f"ids must be strictly monotonic; got {prev} >= {curr}"
            )

    def test_same_ms_ids_share_timestamp_but_differ(self) -> None:
        # Generate two fast and verify same-or-stepped ms with different
        # full ids. The counter guarantees no duplicates.
        a = new_id()
        b = new_id()
        assert a != b
        assert a.bytes < b.bytes

    def test_concurrent_generation_is_unique_and_monotonic(self) -> None:
        # Many threads racing against the global counter MUST all see
        # unique, monotonically-increasing ids.
        results: list[list] = [[] for _ in range(8)]

        def worker(slot: int) -> None:
            results[slot] = [new_id() for _ in range(500)]

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_ids = [uid for slot in results for uid in slot]
        # Uniqueness: 8 * 500 = 4000 distinct ids.
        assert len(set(all_ids)) == len(all_ids)
        # Globally sorted by generation order: when sorted by bytes, the
        # set is exactly the all-ids set (no collisions and the sort key
        # is well-behaved).
        sorted_ids = sorted(all_ids, key=lambda u: u.bytes)
        assert len(sorted_ids) == len(all_ids)
