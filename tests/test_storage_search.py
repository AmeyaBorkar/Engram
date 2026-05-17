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


# --- vector-index concurrency ---------------------------------------------


def test_vector_index_concurrent_rebuild_coalesces(tmp_path: object) -> None:
    """A burst of searches after a write should produce one rebuild,
    not one per searcher.

    Wraps `_rebuild_shard` in a counter to verify the rebuild-in-progress
    flag actually deduplicates concurrent rebuilders.
    """
    import threading
    from pathlib import Path as _Path

    from engram.schemas import Embedding, ItemKind
    from engram.storage import _vector_index as vi
    from engram.storage import SqliteStorage as _SS

    p = _Path(str(tmp_path)) / "vi-race.db"
    backend = _SS(p)
    backend.initialize()
    try:
        # Seed a bunch of event embeddings so the rebuild is non-trivial.
        for i in range(30):
            e = Event(content=f"e{i}")
            backend.insert_event(e)
            backend.insert_embedding(
                Embedding(
                    item_id=e.id,
                    item_kind=ItemKind.EVENT,
                    model="m",
                    dim=2,
                    vector=tuple(_normalize([1.0, float(i) / 30.0])),
                )
            )
        # Warm up cache so the first thread doesn't pay for the bootstrap.
        backend.search_event_embeddings(_normalize([1.0, 0.0]), k=5, model="m")
        # Mark dirty so the next search wave must rebuild.
        backend._vector_index.mark_dirty(kind=ItemKind.EVENT.value, model="m")

        rebuild_calls: list[int] = []
        orig = vi._rebuild_shard

        def counting_rebuild(*args: object, **kwargs: object) -> None:
            rebuild_calls.append(1)
            # Sleep a tiny bit so racing threads have time to converge
            # on the rebuild-in-progress wait path.
            import time

            time.sleep(0.05)
            orig(*args, **kwargs)  # type: ignore[arg-type]

        # Monkeypatch the module-level function the shard uses.
        vi._rebuild_shard = counting_rebuild  # type: ignore[assignment]
        try:
            errors: list[BaseException] = []

            def searcher() -> None:
                try:
                    backend.search_event_embeddings(
                        _normalize([1.0, 0.0]), k=5, model="m"
                    )
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=searcher) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert errors == []
            # Only one rebuild should have run despite 8 racing searchers.
            assert len(rebuild_calls) == 1, (
                f"expected 1 rebuild, got {len(rebuild_calls)} — "
                "rebuild-in-progress flag is not coalescing"
            )
        finally:
            vi._rebuild_shard = orig  # type: ignore[assignment]
    finally:
        backend.close()
