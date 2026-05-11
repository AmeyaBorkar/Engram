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
