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
    return json.dumps(metadata, separators=(",", ":"), sort_keys=True)


def loads_metadata(value: str) -> dict[str, object]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"metadata is not a JSON object: {type(parsed).__name__}")
    return parsed
