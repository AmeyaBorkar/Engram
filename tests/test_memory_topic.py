"""Tests for the Level.TOPIC layer and the extended hierarchy routing."""

from __future__ import annotations

import pytest

from engram import Memory, SqliteStorage
from engram.providers._fake import FakeEmbedder
from engram.retrieve._engine import _GENERALIZATION_LEVELS, _LEVEL_PRIORITY
from engram.schemas import Level


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=8))


class TestRecordTopic:
    def test_creates_topic_item(self, memory: Memory) -> None:
        ev = memory.observe("evidence event")
        topic = memory.record_topic("a topic about X", [ev.id])
        assert topic.level is Level.TOPIC
        # Provenance is wired.
        supporting = memory.storage.get_supporting_events(topic.id)
        assert [e.id for e in supporting] == [ev.id]

    def test_requires_supporting_events(self, memory: Memory) -> None:
        with pytest.raises(ValueError, match="supporting event"):
            memory.record_topic("a topic", [])

    def test_tenant_propagates(self, storage: SqliteStorage) -> None:
        memory = Memory(
            storage=storage,
            embedder=FakeEmbedder(dim=8),
            tenant_id="acme",
        )
        ev = memory.observe("evidence")
        topic = memory.record_topic("topic", [ev.id])
        assert topic.tenant_id == "acme"


class TestHierarchyRouting:
    def test_generalization_levels_include_new_layers(self) -> None:
        assert Level.TOPIC in _GENERALIZATION_LEVELS
        assert Level.PREFERENCE in _GENERALIZATION_LEVELS
        assert Level.GLOBAL in _GENERALIZATION_LEVELS
        assert Level.SUMMARY in _GENERALIZATION_LEVELS
        assert Level.ABSTRACTION in _GENERALIZATION_LEVELS

    def test_level_priority_ordering(self) -> None:
        """Specific-over-general at score ties.

        EVENT(0) < SUMMARY(1) < TOPIC(2) < PREFERENCE(3) <
        ABSTRACTION(4) < GLOBAL(5)
        """
        assert _LEVEL_PRIORITY[Level.EVENT] < _LEVEL_PRIORITY[Level.SUMMARY]
        assert _LEVEL_PRIORITY[Level.SUMMARY] < _LEVEL_PRIORITY[Level.TOPIC]
        assert _LEVEL_PRIORITY[Level.TOPIC] < _LEVEL_PRIORITY[Level.PREFERENCE]
        assert _LEVEL_PRIORITY[Level.PREFERENCE] < _LEVEL_PRIORITY[Level.ABSTRACTION]
        assert _LEVEL_PRIORITY[Level.ABSTRACTION] < _LEVEL_PRIORITY[Level.GLOBAL]


class TestRetrieveSurfacesNewLevels:
    def test_topic_item_surfaces_in_retrieve(self, memory: Memory) -> None:
        """A planted Level.TOPIC item with a matching embedding shows
        up in default retrieve results."""
        ev = memory.observe("evidence")
        memory.record_topic("user is interested in Python", [ev.id])
        results = memory.retrieve(
            "user is interested in Python",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
        )
        levels = {r.level for r in results}
        assert Level.TOPIC in levels

    def test_global_item_surfaces_in_retrieve(self, memory: Memory) -> None:
        memory.set_user_state("the user is a Python dev based in NYC")
        results = memory.retrieve(
            "the user is a Python dev based in NYC",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
        )
        levels = {r.level for r in results}
        assert Level.GLOBAL in levels

    def test_preference_item_surfaces_in_retrieve(self, memory: Memory) -> None:
        memory.record_preference("I love Python")
        results = memory.retrieve(
            "I love Python",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
        )
        levels = {r.level for r in results}
        assert Level.PREFERENCE in levels
