"""Tests for `Storage.search_event_embeddings`."""

from __future__ import annotations

import math

import pytest

from engram.schemas import Embedding, Event, ItemKind
from engram.storage import SqliteStorage


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def _store_event_with_vec(
    storage: SqliteStorage, content: str, vec: list[float], model: str = "m"
) -> Event:
    event = Event(content=content)
    storage.insert_event(event)
    storage.insert_embedding(
        Embedding(
            item_id=event.id,
            item_kind=ItemKind.EVENT,
            model=model,
            dim=len(vec),
            vector=tuple(_normalize(vec)),
        )
    )
    return event


def test_search_returns_empty_when_no_embeddings(storage: SqliteStorage) -> None:
    out = storage.search_event_embeddings([0.5, 0.5], k=10, model="m")
    assert out == []


def test_search_top_k_orders_by_cosine(storage: SqliteStorage) -> None:
    a = _store_event_with_vec(storage, "alpha", [1.0, 0.0])
    b = _store_event_with_vec(storage, "beta", [0.0, 1.0])
    c = _store_event_with_vec(storage, "gamma", [0.7, 0.7])

    query = _normalize([1.0, 0.0])
    results = storage.search_event_embeddings(query, k=3, model="m")
    ids = [r[0] for r in results]
    # alpha is closest, gamma next, beta last
    assert ids[0] == a.id
    assert ids[1] == c.id
    assert ids[2] == b.id


def test_search_returns_at_most_k(storage: SqliteStorage) -> None:
    for i in range(20):
        _store_event_with_vec(storage, f"e{i}", [1.0, 0.0])
    query = _normalize([1.0, 0.0])
    results = storage.search_event_embeddings(query, k=5, model="m")
    assert len(results) == 5


def test_search_handles_k_larger_than_corpus(storage: SqliteStorage) -> None:
    _store_event_with_vec(storage, "only", [1.0, 0.0])
    results = storage.search_event_embeddings(_normalize([1.0, 0.0]), k=50, model="m")
    assert len(results) == 1


def test_search_filters_by_model(storage: SqliteStorage) -> None:
    _store_event_with_vec(storage, "with-m", [1.0, 0.0], model="m")
    _store_event_with_vec(storage, "with-other", [1.0, 0.0], model="other")
    results = storage.search_event_embeddings(_normalize([1.0, 0.0]), k=10, model="m")
    contents = [r[1] for r in results]
    assert contents == ["with-m"]


def test_search_returns_score_close_to_1_for_identical_vec(storage: SqliteStorage) -> None:
    _store_event_with_vec(storage, "x", [0.6, 0.8])
    results = storage.search_event_embeddings(_normalize([0.6, 0.8]), k=1, model="m")
    assert results[0][2] == pytest.approx(1.0, abs=1e-5)


def test_search_rejects_dim_mismatch(storage: SqliteStorage) -> None:
    _store_event_with_vec(storage, "x", [0.6, 0.8])  # dim=2
    with pytest.raises(ValueError, match="does not match"):
        storage.search_event_embeddings([0.1, 0.2, 0.3], k=1, model="m")


def test_search_rejects_invalid_k(storage: SqliteStorage) -> None:
    with pytest.raises(ValueError, match="k must be"):
        storage.search_event_embeddings([0.5, 0.5], k=0, model="m")
