"""Tests for storage serialization helpers (`engram.storage._serialize`)."""

from __future__ import annotations

import pytest

from engram.storage._serialize import (
    loads_metadata,
    pack_vector,
    unpack_vector,
)


def test_pack_unpack_vector_roundtrip() -> None:
    vec = (0.1, -2.0, 3.14, 1e-6)
    blob = pack_vector(vec)
    assert len(blob) == 4 * len(vec)
    out = unpack_vector(blob, len(vec))
    assert out == pytest.approx(vec)


def test_unpack_vector_rejects_size_mismatch() -> None:
    blob = pack_vector([0.1, 0.2])
    with pytest.raises(ValueError, match="blob size"):
        unpack_vector(blob, dim=4)


def test_loads_metadata_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="not a JSON object"):
        loads_metadata("[1, 2, 3]")


def test_loads_metadata_accepts_object() -> None:
    assert loads_metadata('{"k": "v"}') == {"k": "v"}
