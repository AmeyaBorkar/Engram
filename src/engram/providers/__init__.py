"""Provider abstraction.

Stage 2 lays down the embedding and chat-completion abstraction that
consolidation, retrieval, and the bench harness all share. The shape:

  - `EmbeddingProvider` / `ChatProvider` — protocols for sync + async surfaces.
  - `FakeProvider` — deterministic, hash-based; used by every unit test.
  - `OpenAIEmbedder` / `OpenAIChat` — concrete adapters (extras: `[openai]`).
  - `AnthropicChat` — concrete adapter (extras: `[anthropic]`).
  - `Retry`, `Cache`, `Redactor`, `Batcher` — cross-cutting wrappers.

Stage 2 is implemented one primitive at a time; this `__init__` re-exports
the public surface as it lands.
"""

from engram.providers._retry import Retry

__all__ = ["Retry"]
