"""Deterministic fake providers.

Used by every unit test that needs an embedding or chat call. No network,
no API keys, no nondeterminism: every call is a pure function of its input.

The hash-based embedder produces unit-norm vectors derived from SHA-256 of
the input text, expanded to the requested dimension via stretched output.
The scripted chat returns either a configured response (keyed by the
content hash of the last user message) or a default fallback.

For tests that need to verify "the chat was called," the default fallback
includes the input hash so the assertion can pin to a specific input.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence

from engram.providers._cache import content_hash
from engram.providers._message import Message


class FakeEmbedder:
    """Hash-based deterministic embeddings.

    Same `text` always produces the same vector. Different `text` strings
    produce vectors that are unit-norm and pseudo-uniformly distributed in
    direction (within the limits of SHA-256 as a PRG).
    """

    name: str = "fake-embed"

    def __init__(self, dim: int = 128, model: str = "fake-sha256") -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts)

    def manifest_hash(self) -> str:
        return f"fake-embed/sha256/dim={self.dim}/v1"

    def _embed_one(self, text: str) -> list[float]:
        # Stretch SHA-256 output until we have `dim * 2` bytes (one int16 per dim).
        # Length-prefix the text so an input ending in ":N" can't collide
        # with a different input whose counter starts mid-stretch (e.g.
        # text="foo:0\xff" + counter=0 vs text="foo:0" + counter=0xff).
        needed = self.dim * 2
        blob = b""
        counter = 0
        encoded = text.encode("utf-8")
        prefix = f"L={len(encoded)}|"
        while len(blob) < needed:
            blob += hashlib.sha256(f"{prefix}{text}:{counter}".encode()).digest()
            counter += 1
        blob = blob[:needed]

        shorts = struct.unpack(f"<{self.dim}h", blob)
        floats = [s / 32768.0 for s in shorts]

        norm = math.sqrt(sum(f * f for f in floats))
        if norm == 0.0:
            return floats
        return [f / norm for f in floats]


class FakeChat:
    """Scripted chat. Looks up a response by content-hash of the last user message.

    Pass `scripts={hash_of_input: response, ...}` to wire specific replies.
    The default fallback embeds the input hash so a test can assert which
    input triggered it without running the model.
    """

    name: str = "fake-chat"

    def __init__(
        self,
        scripts: Mapping[str, str] | None = None,
        *,
        default: str | None = None,
        model: str = "fake-scripted",
    ) -> None:
        self._scripts: dict[str, str] = dict(scripts) if scripts else {}
        self._default = default
        self.model = model

    def chat(self, messages: Sequence[Message]) -> str:
        return self._reply(messages)

    async def achat(self, messages: Sequence[Message]) -> str:
        return self._reply(messages)

    def manifest_hash(self) -> str:
        # Stable hash of (model, scripts) so two FakeChats with the same
        # configuration produce identical manifest hashes.
        h = hashlib.sha256()
        h.update(self.model.encode("utf-8"))
        h.update(b"\x00")
        for key in sorted(self._scripts):
            h.update(key.encode("utf-8"))
            h.update(b"\x01")
            h.update(self._scripts[key].encode("utf-8"))
            h.update(b"\x02")
        if self._default is not None:
            h.update(self._default.encode("utf-8"))
        return f"fake-chat/{h.hexdigest()[:16]}"

    def _reply(self, messages: Sequence[Message]) -> str:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        if last_user is None:
            return self._default or "[fake: no user message]"
        key = content_hash(last_user.content)
        if key in self._scripts:
            return self._scripts[key]
        if self._default is not None:
            return self._default
        return f"[fake: input_hash={key[:12]}]"
