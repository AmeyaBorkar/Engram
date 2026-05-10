"""Smoke test: the package imports and the public surface is reachable."""

import engram


def test_memory_is_exported() -> None:
    assert engram.Memory is not None


def test_memory_is_instantiable() -> None:
    from engram.providers import FakeEmbedder

    storage = engram.SqliteStorage(":memory:")
    storage.initialize()
    try:
        m = engram.Memory(storage=storage, embedder=FakeEmbedder(dim=8))
        assert isinstance(m, engram.Memory)
    finally:
        storage.close()


def test_version_is_set() -> None:
    assert isinstance(engram.__version__, str)
    assert engram.__version__ != ""
