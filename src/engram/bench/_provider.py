"""Bench-harness provider abstraction.

The harness records `name` and `manifest_hash` on every run; suites that
need real model calls duck-type into the bundled `embedder` and `chat`.

Stage 1 shipped a stub `FakeProvider` so the harness could run before
the Stage 2 abstraction landed. This module now wraps the real Stage 2
fakes (`engram.providers.FakeEmbedder` / `FakeChat`) so suites have
deterministic embedding and chat surfaces available without going to a
network.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from engram.providers import (
    ChatProvider,
    EmbeddingProvider,
    FakeChat,
    FakeEmbedder,
)


@runtime_checkable
class Provider(Protocol):
    """Minimum surface the bench harness records for every run."""

    name: str

    def manifest_hash(self) -> str:
        """Stable identifier of this provider's configured behavior."""


class FakeProvider:
    """Harness fake: bundles a `FakeEmbedder` and a `FakeChat`.

    Suites that need an embedding provider use `provider.embedder`; suites
    that need a chat provider use `provider.chat`. The bench harness only
    looks at `name` and `manifest_hash()` itself.
    """

    name: str = "fake"

    def __init__(
        self,
        *,
        dim: int = 128,
        chat_scripts: dict[str, str] | None = None,
        chat_default: str | None = None,
    ) -> None:
        self.embedder: EmbeddingProvider = FakeEmbedder(dim=dim)
        self.chat: ChatProvider = FakeChat(scripts=chat_scripts, default=chat_default)

    def manifest_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.embedder.manifest_hash().encode("utf-8"))
        h.update(b"|")
        h.update(self.chat.manifest_hash().encode("utf-8"))
        return f"fake/{h.hexdigest()[:16]}"
