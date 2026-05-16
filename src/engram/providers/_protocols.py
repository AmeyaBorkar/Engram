"""Provider protocols.

`EmbeddingProvider` and `ChatProvider` are the two surfaces every
implementation honors.  Both expose sync and async paths; the async
path is consumed by `Memory.aretrieve` / `aconsolidate` etc.

The `name`, `model`, and `manifest_hash()` triple is what benchmark
manifests record; two providers compare equal by their hash, so swapping a
provider with the same hash should yield reproducible runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from engram.providers._message import Message


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns a list of texts into a list of dense vectors."""

    name: str
    model: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def aembed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def manifest_hash(self) -> str: ...


@runtime_checkable
class ChatProvider(Protocol):
    """Turns a list of messages into an assistant reply."""

    name: str
    model: str

    def chat(self, messages: Sequence[Message]) -> str: ...
    async def achat(self, messages: Sequence[Message]) -> str: ...

    def manifest_hash(self) -> str: ...
