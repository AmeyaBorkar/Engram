"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from engram.storage import SqliteStorage


@pytest.fixture
def storage() -> Iterator[SqliteStorage]:
    """In-memory SQLite storage. Fast; recreated per test."""
    backend = SqliteStorage(":memory:")
    backend.initialize()
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture
def disk_storage(tmp_path: Path) -> Iterator[SqliteStorage]:
    """On-disk SQLite storage in a tmp dir; survives reopen for migration tests."""
    backend = SqliteStorage(tmp_path / "engram.db")
    backend.initialize()
    try:
        yield backend
    finally:
        backend.close()
