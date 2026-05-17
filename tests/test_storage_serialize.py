"""Tests for storage serialization helpers (`engram.storage._serialize`)."""

from __future__ import annotations

import pytest

from engram.storage._serialize import (
    dumps_metadata,
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


def test_dumps_metadata_roundtrips_serializable() -> None:
    payload = {"k": "v", "n": 1, "f": 0.5, "bool": True, "null": None, "list": [1, 2]}
    encoded = dumps_metadata(payload)
    assert loads_metadata(encoded) == payload


def test_dumps_metadata_rejects_non_serializable() -> None:
    """Regression for M-14: helper validates serializability up-front.

    Previously a non-JSON-serializable value (e.g. a `set`, a Pydantic
    model, a dataclass) raised a bare `TypeError` deep inside `json.dumps`
    at the storage boundary — losing any context about which row/key
    caused the failure.  The helper now wraps the raise as `ValueError`
    with a clear message.
    """
    with pytest.raises(ValueError, match="metadata not JSON-serializable"):
        # Sets aren't JSON-serializable.
        dumps_metadata({"bad": {1, 2, 3}})
