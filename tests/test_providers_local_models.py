"""Tests for `engram.providers.local.RECOMMENDED_MODELS`.

We don't load the models in CI (they're huge); we just verify the
catalog shape is correct and that LocalEmbedder accepts any of the
listed model strings at type level.
"""

from __future__ import annotations

import pytest


def test_recommended_models_catalog() -> None:
    """The catalog includes the Tier 4 / E.15 top embedders plus the
    current default."""
    from engram.providers.local import RECOMMENDED_MODELS

    expected = {
        "nvidia/NV-Embed-v2",
        "dunzhang/stella_en_1.5B_v5",
        "BAAI/bge-large-en-v1.5",
        "mixedbread-ai/mxbai-embed-large-v1",
        "BAAI/bge-m3",
        "BAAI/bge-base-en-v1.5",
        "intfloat/e5-large-v2",
        "nomic-ai/nomic-embed-text-v1.5",
    }
    assert set(RECOMMENDED_MODELS) >= expected


def test_each_entry_has_required_fields() -> None:
    from engram.providers.local import RECOMMENDED_MODELS

    for model_id, entry in RECOMMENDED_MODELS.items():
        assert "dim" in entry, f"{model_id}: missing 'dim'"
        assert "params" in entry, f"{model_id}: missing 'params'"
        assert "notes" in entry, f"{model_id}: missing 'notes'"
        assert isinstance(entry["dim"], int)
        assert entry["dim"] > 0
        assert isinstance(entry["params"], str)
        assert isinstance(entry["notes"], str)


def test_dims_are_sensible() -> None:
    """Sanity: catalog dims fall within the range of real embedding models."""
    from engram.providers.local import RECOMMENDED_MODELS

    for model_id, entry in RECOMMENDED_MODELS.items():
        dim = entry["dim"]
        assert isinstance(dim, int)
        assert 64 <= dim <= 8192, f"{model_id}: dim={dim} out of expected range"


@pytest.mark.slow
def test_local_embedder_loads_bge_large() -> None:
    """Smoke: the default model loads on this machine. Slow because
    sentence-transformers has to import torch and download/cache the
    model on first run."""
    from engram.providers.local import LocalEmbedder

    embedder = LocalEmbedder(device="cpu")
    out = embedder.embed(["hello"])
    assert len(out) == 1
    assert len(out[0]) == embedder.dim


# --- lazy load + unload (M-82 / M-95) -------------------------------------


class _FakeST:
    """Minimal stub of sentence_transformers.SentenceTransformer.

    Implements just the surface LocalEmbedder uses. Tracks load count
    via a class-level counter so tests can assert lazy-load behavior."""

    load_count = 0

    def __init__(self, model: str, device: str | None = None) -> None:
        type(self).load_count += 1
        self._model = model
        self._device = device
        # bge-large native dim, matches RECOMMENDED_MODELS catalog.
        self._dim = 1024
        self._halved = False

    def half(self) -> None:
        self._halved = True

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str], **_: object) -> object:
        import numpy as np

        # Deterministic toy embeddings; values don't matter for these tests.
        return np.array([[float(len(t)) / 100.0] * self._dim for t in texts])


def _patch_st(monkeypatch: pytest.MonkeyPatch) -> type[_FakeST]:
    """Patch the module-level lazy import path."""
    _FakeST.load_count = 0
    import sentence_transformers as st_mod

    monkeypatch.setattr(st_mod, "SentenceTransformer", _FakeST)
    return _FakeST


def test_local_embedder_does_not_load_on_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _patch_st(monkeypatch)
    from engram.providers.local import LocalEmbedder

    # bge-large-en-v1.5 is in the catalog, so the dim is known without
    # touching the model.
    e = LocalEmbedder(device="cpu")
    assert fake.load_count == 0
    assert e.dim == 1024
    assert e.is_loaded is False


def test_local_embedder_loads_on_first_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _patch_st(monkeypatch)
    from engram.providers.local import LocalEmbedder

    e = LocalEmbedder(device="cpu", cache_size=0)
    _ = e.embed(["hi"])
    assert fake.load_count == 1
    # Second call must not re-load.
    _ = e.embed(["world"])
    assert fake.load_count == 1


def test_local_embedder_unload_releases_and_reloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _patch_st(monkeypatch)
    from engram.providers.local import LocalEmbedder

    e = LocalEmbedder(device="cpu", cache_size=0)
    _ = e.embed(["a"])
    assert e.is_loaded is True
    e.unload()
    assert e.is_loaded is False
    _ = e.embed(["b"])
    assert fake.load_count == 2


def test_local_embedder_unload_clears_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_st(monkeypatch)
    from engram.providers.local import LocalEmbedder

    e = LocalEmbedder(device="cpu", cache_size=16)
    _ = e.embed(["a", "b"])
    assert e.cache is not None
    assert len(e.cache) == 2
    e.unload()
    assert len(e.cache) == 0


def test_local_embedder_symmetric_query_shares_doc_cache_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On symmetric models (no `_QUERY_ENCODE_KWARGS`), embed_query and
    embed share the same cache slot for a given text. Warming one
    pre-warms the other (M-174)."""
    fake = _patch_st(monkeypatch)
    from engram.providers.local import LocalEmbedder

    e = LocalEmbedder(device="cpu", cache_size=16)
    # Warm doc cache with "hi".
    _ = e.embed(["hi"])
    encode_calls_after_doc = fake.load_count
    cache = e.cache
    assert cache is not None
    # First time -- doc encode runs once.
    assert cache.hits == 0
    # embed_query for the same text should hit the cache without
    # encoding again.
    _ = e.embed_query("hi")
    assert cache.hits >= 1
    # The underlying model load count must not have grown.
    assert fake.load_count == encode_calls_after_doc


# --- Matryoshka truncation (M-176) ----------------------------------------


def test_local_embedder_target_dim_truncates_and_renormalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_st(monkeypatch)
    from engram.providers.local import LocalEmbedder

    e = LocalEmbedder(device="cpu", target_dim=256, cache_size=0)
    assert e.dim == 256
    out = e.embed(["hello"])
    assert len(out) == 1
    assert len(out[0]) == 256
    # L2-normalized: sum-of-squares ~ 1.
    sumsq = sum(x * x for x in out[0])
    assert abs(sumsq - 1.0) < 1e-6


def test_local_embedder_target_dim_no_truncation_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_st(monkeypatch)
    from engram.providers.local import LocalEmbedder

    # target_dim == native_dim: no truncation.
    e = LocalEmbedder(device="cpu", target_dim=1024, cache_size=0)
    assert e.dim == 1024
    out = e.embed(["x"])
    assert len(out[0]) == 1024


def test_local_embedder_target_dim_rejects_out_of_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_st(monkeypatch)
    from engram.providers.local import LocalEmbedder

    with pytest.raises(ValueError, match="target_dim"):
        LocalEmbedder(device="cpu", target_dim=0)
    with pytest.raises(ValueError, match="target_dim"):
        LocalEmbedder(device="cpu", target_dim=99999)


def test_local_embedder_target_dim_changes_manifest_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_st(monkeypatch)
    from engram.providers.local import LocalEmbedder

    a = LocalEmbedder(device="cpu", cache_size=0)
    b = LocalEmbedder(device="cpu", target_dim=256, cache_size=0)
    assert a.manifest_hash() != b.manifest_hash()


def test_local_embedder_non_matryoshka_warns_on_truncation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_st(monkeypatch)
    from engram.providers.local import LocalEmbedder

    # bge-large is not Matryoshka.
    with caplog.at_level("WARNING", logger="engram.providers.local"):
        LocalEmbedder(device="cpu", target_dim=256, cache_size=0)
    assert any("non-Matryoshka" in rec.message for rec in caplog.records)
