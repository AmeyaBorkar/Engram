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


# --- dumps_metadata error surface -----------------------------------------


def test_dumps_metadata_raises_value_error_for_non_serializable() -> None:
    """Non-JSON-serializable values surface as ValueError (not TypeError)
    so callers can catch the storage-boundary error class uniformly."""
    from engram.storage._serialize import dumps_metadata

    class _Opaque:
        pass

    with pytest.raises(ValueError, match="not JSON-serializable"):
        dumps_metadata({"obj": _Opaque()})


def test_dumps_metadata_error_names_offending_key() -> None:
    from engram.storage._serialize import dumps_metadata

    class _Opaque:
        pass

    with pytest.raises(ValueError, match=r"key 'bad'"):
        dumps_metadata({"ok": "v", "bad": _Opaque()})


def test_dumps_metadata_round_trip() -> None:
    from engram.storage._serialize import dumps_metadata

    blob = dumps_metadata({"a": 1, "b": [1, 2, 3], "c": {"nested": True}})
    assert loads_metadata(blob) == {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
