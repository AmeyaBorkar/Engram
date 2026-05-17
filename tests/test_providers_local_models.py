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


# ---------------------------------------------------------------------------
# M-95 / M-174: unload + symmetric-model embed_query fallback
# ---------------------------------------------------------------------------


def _build_stub_embedder(model_name: str = "BAAI/bge-large-en-v1.5"):
    """Build a LocalEmbedder against a stubbed SentenceTransformer.

    We don't want to download a 1.3 GiB model to test
    cache-fallback / unload mechanics — instead patch the constructor's
    SentenceTransformer reference to a stub that returns deterministic
    numpy arrays.
    """
    import numpy as np

    from engram.providers import local as local_module

    class _StubST:
        def __init__(self, model: str, device: str = "cpu") -> None:
            self._model = model
            self._dim = 8

        def get_sentence_embedding_dimension(self) -> int:
            return self._dim

        def encode(
            self,
            texts,
            *,
            batch_size: int = 64,
            convert_to_numpy: bool = True,
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
            prompt: str | None = None,
            prompt_name: str | None = None,
        ):
            # Stable deterministic output: hash text -> first 8 bytes
            # converted to floats.  No prompt distinction here on
            # symmetric models, exactly matching the real ST behavior
            # we're trying to model (M-174 fallback only applies when
            # query and doc paths produce the SAME vector).
            out = np.zeros((len(texts), self._dim), dtype="float32")
            for i, t in enumerate(texts):
                seed = hash(t) & 0xFFFFFFFF
                rng = np.random.RandomState(seed)
                out[i] = rng.random(self._dim).astype("float32")
            return out

        def half(self) -> None:  # pragma: no cover - half not used by tests
            return None

    return local_module, _StubST


def test_local_embedder_unload_drops_model_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_module, stub_cls = _build_stub_embedder()
    monkeypatch.setattr(local_module, "_SentenceTransformer", stub_cls)
    monkeypatch.setattr(local_module, "_detect_device", lambda: "cpu")
    embedder = local_module.LocalEmbedder(model="BAAI/bge-large-en-v1.5", cache_size=32)
    # Warm the cache.
    embedder.embed(["hello"])
    assert embedder.cache is not None
    assert len(embedder.cache) >= 1
    embedder.unload()
    # Model handle is dropped.
    assert not hasattr(embedder, "_st") or embedder._st is None  # noqa: SLF001
    # Cache is cleared and detached.
    assert embedder.cache is None


def test_local_embedder_unload_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_module, stub_cls = _build_stub_embedder()
    monkeypatch.setattr(local_module, "_SentenceTransformer", stub_cls)
    monkeypatch.setattr(local_module, "_detect_device", lambda: "cpu")
    embedder = local_module.LocalEmbedder(model="BAAI/bge-large-en-v1.5")
    embedder.unload()
    # Second unload must not raise.
    embedder.unload()


def test_local_embedder_context_manager_unloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_module, stub_cls = _build_stub_embedder()
    monkeypatch.setattr(local_module, "_SentenceTransformer", stub_cls)
    monkeypatch.setattr(local_module, "_detect_device", lambda: "cpu")
    with local_module.LocalEmbedder(model="BAAI/bge-large-en-v1.5") as e:
        e.embed(["x"])
    assert e.cache is None


def test_local_embedder_embed_query_falls_back_to_doc_cache_for_symmetric_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric models (no `_QUERY_ENCODE_KWARGS` entry) must satisfy
    `embed_query("X")` from a prior `embed(["X"])` cache hit.  Saves
    one GPU encode per query when the same text was previously
    ingested as a document.
    """
    local_module, stub_cls = _build_stub_embedder()
    monkeypatch.setattr(local_module, "_SentenceTransformer", stub_cls)
    monkeypatch.setattr(local_module, "_detect_device", lambda: "cpu")
    embedder = local_module.LocalEmbedder(
        model="BAAI/bge-large-en-v1.5", cache_size=32
    )

    # First, embed as document.
    docs = embedder.embed(["alpha"])
    encode_calls: list[int] = []

    real_encode = embedder._encode  # noqa: SLF001

    def traced_encode(texts, *, extra=None):
        encode_calls.append(len(texts))
        return real_encode(texts, extra=extra)

    embedder._encode = traced_encode  # type: ignore[method-assign]  # noqa: SLF001

    # Calling embed_query("alpha") on a symmetric model should NOT
    # trigger a fresh encode — the doc-side cache slot is reused.
    q = embedder.embed_query("alpha")
    assert q == docs[0]
    assert encode_calls == []  # no GPU encode on the fallback path


def test_local_embedder_embed_query_keeps_strict_split_for_asymmetric_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asymmetric models (stella, e5, ...) must NOT cross query / doc
    cache slots — the vectors differ between paths."""
    local_module, stub_cls = _build_stub_embedder()
    monkeypatch.setattr(local_module, "_SentenceTransformer", stub_cls)
    monkeypatch.setattr(local_module, "_detect_device", lambda: "cpu")
    # Asymmetric model — entry in _QUERY_ENCODE_KWARGS so
    # `_is_symmetric()` returns False.
    embedder = local_module.LocalEmbedder(
        model="intfloat/e5-large-v2", cache_size=32
    )

    embedder.embed(["alpha"])
    encode_calls: list[int] = []
    real_encode = embedder._encode  # noqa: SLF001

    def traced_encode(texts, *, extra=None):
        encode_calls.append(len(texts))
        return real_encode(texts, extra=extra)

    embedder._encode = traced_encode  # type: ignore[method-assign]  # noqa: SLF001

    embedder.embed_query("alpha")
    # Asymmetric path: doc-cache hit must NOT serve query slot.
    assert encode_calls == [1]
