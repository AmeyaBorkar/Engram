"""Internal serialization helpers shared by storage backends."""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from datetime import datetime


def pack_vector(vector: Sequence[float]) -> bytes:
    """Pack floats as little-endian float32 bytes."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes, dim: int) -> tuple[float, ...]:
    """Unpack little-endian float32 bytes into a tuple of `dim` floats."""
    expected = dim * 4
    if len(blob) != expected:
        raise ValueError(f"blob size {len(blob)} != dim*4 ({expected})")
    return struct.unpack(f"<{dim}f", blob)


def iso(dt: datetime) -> str:
    """ISO-8601 string for a datetime. Used for SQLite TEXT timestamps."""
    return dt.isoformat()


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def dumps_metadata(metadata: dict[str, object]) -> str:
    """Serialize `metadata` to a canonical compact JSON string.

    Raises `ValueError` with a clear message if the dict contains a
    value `json.dumps` can't encode (objects with `__dict__`, sets,
    bytes, etc.).  Without this wrapper the storage write would raise
    a bare `TypeError: Object of type X is not JSON serializable`
    far from the call site — the typed `ValueError` is easier to
    catch at the Memory API boundary and tells the caller which
    key is the offender.
    """
    try:
        return json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    except TypeError as exc:
        # json.dumps's error message includes the offending type but
        # not the key.  Find a problematic key by re-serializing each
        # value alone; on Python's stdlib json this is O(n) over the
        # dict, but metadata is small (a few keys).
        bad_key: str | None = None
        for k, v in metadata.items():
            try:
                json.dumps(v)
            except TypeError:
                bad_key = k
                break
        if bad_key is not None:
            raise ValueError(
                f"metadata value at key {bad_key!r} is not JSON-serializable: {exc}"
            ) from exc
        raise ValueError(f"metadata is not JSON-serializable: {exc}") from exc


def loads_metadata(value: str) -> dict[str, object]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"metadata is not a JSON object: {type(parsed).__name__}")
    return parsed
