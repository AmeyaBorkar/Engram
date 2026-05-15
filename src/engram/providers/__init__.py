"""Provider abstraction.

This package defines the embedding and chat-completion abstraction that
consolidation, retrieval, and the bench harness all share. The shape:

  - `EmbeddingProvider` / `ChatProvider` — protocols for sync + async surfaces.
  - `FakeProvider` — deterministic, hash-based; used by every unit test.
  - `OpenAIEmbedder` / `OpenAIChat` — concrete adapters (extras: `[openai]`).
  - `AnthropicChat` — concrete adapter (extras: `[anthropic]`).
  - `Retry`, `Cache`, `Redactor`, `Batcher` — cross-cutting wrappers
    that compose around any provider.

Cross-cutting wrappers (composition example)
--------------------------------------------

The wrappers are intentionally orthogonal so callers can layer the
behaviors they need without inheritance gymnastics. A reasonable
production stack looks like::

    from engram.providers import Redactor, Retry, content_hash
    from engram.providers.openai import OpenAIChat

    chat = OpenAIChat(api_key=...)
    retry = Retry(max_attempts=5, base_delay=0.5, jitter=True)
    redactor = Redactor.default()

    def safe_chat(messages):
        # Retry transient SDK errors; redact what we log before raising.
        try:
            return retry.call(chat.chat, messages)
        except Exception as exc:
            log.error("chat failed: %s", redactor.redact(repr(exc)))
            raise

`Cache` (in-memory LRU, thread-safe) and `Batcher` (coalesce concurrent
submits) compose at the same layer. The persistent-disk variant lives
in `engram.providers._disk_cache.with_disk_cache(provider, path=...)`
and proxies the same surface as the underlying provider so it slots in
anywhere the original was accepted.

`Redactor.default()` covers provider API keys, Authorization /
x-api-key headers, Bearer tokens (including base64 / JWT shapes),
emails, phones, SSNs, and credit-card-shaped digit groups. Add your
own via `Redactor.from_patterns([...])`.
"""

from engram.providers._batcher import Batcher
from engram.providers._cache import Cache, content_hash
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.providers._message import Message, Role
from engram.providers._protocols import ChatProvider, EmbeddingProvider
from engram.providers._redactor import Redactor
from engram.providers._retry import Retry

__all__ = [
    "Batcher",
    "Cache",
    "ChatProvider",
    "EmbeddingProvider",
    "FakeChat",
    "FakeEmbedder",
    "Message",
    "Redactor",
    "Retry",
    "Role",
    "content_hash",
]
