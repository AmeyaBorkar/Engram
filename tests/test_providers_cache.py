"""Tests for `engram.providers.Cache` and `content_hash`."""

from __future__ import annotations

import pytest

from engram.providers import Cache, content_hash


def test_get_missing_returns_none() -> None:
    c: Cache[int] = Cache()
    assert c.get("nope") is None
    assert c.misses == 1
    assert c.hits == 0


def test_set_then_get_hits() -> None:
    c: Cache[int] = Cache()
    c.set("k", 42)
    assert c.get("k") == 42
    assert c.hits == 1
    assert c.misses == 0


def test_lru_evicts_oldest() -> None:
    c: Cache[int] = Cache(max_size=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    assert "a" not in c
    assert "b" in c
    assert "c" in c


def test_get_promotes_recent_use() -> None:
    c: Cache[int] = Cache(max_size=2)
    c.set("a", 1)
    c.set("b", 2)
    _ = c.get("a")  # bumps a to most-recent
    c.set("c", 3)  # evicts b, not a
    assert "a" in c
    assert "b" not in c


def test_set_updates_existing_value_and_promotes() -> None:
    c: Cache[int] = Cache(max_size=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("a", 99)
    c.set("c", 3)
    assert c.get("a") == 99
    assert "b" not in c


def test_clear_resets_data_and_counters() -> None:
    c: Cache[int] = Cache()
    c.set("a", 1)
    c.get("a")
    c.get("nope")
    c.clear()
    assert len(c) == 0
    assert c.hits == 0
    assert c.misses == 0


def test_hit_rate_with_no_lookups_is_zero() -> None:
    c: Cache[int] = Cache()
    assert c.hit_rate == 0.0


def test_hit_rate_observable() -> None:
    c: Cache[int] = Cache()
    c.set("a", 1)
    c.get("a")  # hit
    c.get("a")  # hit
    c.get("b")  # miss
    assert c.hit_rate == pytest.approx(2 / 3)


def test_max_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_size"):
        Cache(max_size=0)


def test_contains_does_not_count_as_lookup() -> None:
    c: Cache[int] = Cache()
    c.set("a", 1)
    assert "a" in c
    assert "b" not in c
    assert c.hits == 0
    assert c.misses == 0


def test_content_hash_deterministic() -> None:
    assert content_hash("hello") == content_hash("hello")


def test_content_hash_collision_resistant_for_concatenated_inputs() -> None:
    # ("ab", "c") must not collide with ("a", "bc") because of the NUL separator.
    assert content_hash("ab", "c") != content_hash("a", "bc")


def test_content_hash_returns_hex() -> None:
    h = content_hash("x")
    assert len(h) == 64
    assert all(ch in "0123456789abcdef" for ch in h)
