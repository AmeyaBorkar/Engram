"""Prompt-injection corpus tests.

The deterministic `FakeChat` cannot be injected by construction (it does
not "interpret" any input). These tests assert that and document the
test surface so Stage 5's consolidation prompt can plug in directly.

When Stage 5 ships, the same corpus drives a parameterized test against
the real consolidation prompt with a real chat provider, and an
abstraction produced from the corpus must contain none of the forbidden
substrings.
"""

from __future__ import annotations

import pytest

from engram._security import CORPUS, InjectionAttempt
from engram.providers import FakeChat, Message


def test_corpus_is_nonempty() -> None:
    # Pinned to >= 15 so the expanded surface (Phase 0 added Unicode,
    # RTL, base64, multilingual, etc.) cannot silently regress to the
    # 8-entry ASCII-only set that preceded it.  Bumping this floor is
    # encouraged whenever the corpus grows.
    assert len(CORPUS) >= 15


def test_corpus_entries_have_forbidden_substrings() -> None:
    for attempt in CORPUS:
        assert attempt.forbidden_substrings, f"{attempt.name}: must declare forbidden substrings"


def test_corpus_names_are_unique() -> None:
    names = [a.name for a in CORPUS]
    assert len(names) == len(set(names)), "duplicate names in corpus"


@pytest.mark.parametrize("attempt", CORPUS, ids=lambda a: a.name)
def test_fake_chat_does_not_emit_forbidden_substrings(attempt: InjectionAttempt) -> None:
    """The deterministic fake never echoes the attack content; this is the
    base case the real consolidation prompt at Stage 5 must also satisfy."""
    chat = FakeChat()
    reply = chat.chat(
        [
            Message(role="system", content="Summarize the following event."),
            Message(role="user", content=attempt.payload),
        ]
    )
    for forbidden in attempt.forbidden_substrings:
        assert forbidden not in reply, (
            f"FakeChat reply contained forbidden substring {forbidden!r} "
            f"for attempt {attempt.name!r}: {reply!r}"
        )


def test_corpus_payloads_are_distinct() -> None:
    payloads = [a.payload for a in CORPUS]
    assert len(payloads) == len(set(payloads))
