"""Real-provider builders for the bench harness.

The bench harness shipped Stage 1 with `FakeProvider` only, sufficient
for smoke tests. Stage 6's LongMemEval / LoCoMo runs need real
embedding + chat APIs, but the surface is intentionally narrow so each
suite still uses one `Provider` object.

Only OpenAI and Anthropic ship official Python SDKs in our extras.
Moonshot/Kimi (and other OpenAI-compatible endpoints like OpenRouter
and Together) reuse the OpenAI adapter via its `base_url` parameter.
Anthropic ships no embedding model -- if the chat side is Anthropic,
the embedder side falls back to OpenAI by convention.

Naming: the Provider's `name` field flows into the manifest, so two
runs with different (embedder, chat) pairs stay distinguishable in the
SCOREBOARD without anyone having to remember a config blob.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

from engram.providers import ChatProvider, EmbeddingProvider, FakeChat, FakeEmbedder


class _MixedProvider:
    """Bundles one embedder + one chat into a single bench Provider.

    Built by `build_provider` -- no public constructor. The class exists
    so each Provider snapshots its `(embedder.manifest_hash, chat.
    manifest_hash)` pair in `manifest_hash()`, giving us a stable
    fingerprint for reproducibility.
    """

    def __init__(
        self,
        *,
        name: str,
        embedder: EmbeddingProvider,
        chat: ChatProvider,
    ) -> None:
        self.name = name
        self.embedder = embedder
        self.chat = chat

    def manifest_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.embedder.manifest_hash().encode("utf-8"))
        h.update(b"|")
        h.update(self.chat.manifest_hash().encode("utf-8"))
        return f"{self.name}/{h.hexdigest()[:16]}"


def _moonshot_chat(model: str | None) -> ChatProvider:
    from engram.providers.openai import OpenAIChat

    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOONSHOT_API_KEY is not set. Get a key from https://platform.moonshot.ai/"
        )
    return OpenAIChat(
        model=model or "kimi-k2.6",
        api_key=api_key,
        base_url="https://api.moonshot.ai/v1",
    )


def _opencode_api_key() -> str:
    """Resolve the OpenCode platform key.

    OpenCode's Zen and Go plans share the same account-level API key;
    most users have ONE key. We accept several env var names so the
    setup reads naturally regardless of which plan you signed up for:

      OPENCODE_API_KEY     -- primary (works for both Zen and Go).
      OPENCODE_ZEN_API_KEY -- alias, common for Zen-only setups.
      OPENCODE_GO_API_KEY  -- alias, common for Go-only setups.
    """
    for var in ("OPENCODE_API_KEY", "OPENCODE_ZEN_API_KEY", "OPENCODE_GO_API_KEY"):
        value = os.environ.get(var)
        if value:
            return value
    raise RuntimeError(
        "OpenCode API key is not set. Add OPENCODE_API_KEY (or one of "
        "OPENCODE_ZEN_API_KEY / OPENCODE_GO_API_KEY) to your .env. "
        "Get a key from https://opencode.ai/"
    )


def _opencode_zen_chat(model: str | None) -> ChatProvider:
    """OpenCode Zen — multi-model gateway with OpenAI-compatible chat API.

    One API key, access to Claude (haiku/sonnet/opus 4.x), GPT 5.x, and
    Kimi K2.x. No embedding model -- pair with `--embedder openai`.
    Default model is `claude-haiku-4-5` because it's cheap, fast, and
    matches what most LongMemEval-style reports use.
    """
    from engram.providers.openai import OpenAIChat

    return OpenAIChat(
        model=model or "claude-haiku-4-5",
        api_key=_opencode_api_key(),
        base_url="https://opencode.ai/zen/v1",
    )


def _opencode_go_chat(model: str | None) -> ChatProvider:
    """OpenCode Go -- open-weight coding models behind one subscription.

    Same OpenCode account / API key as Zen, but a different endpoint
    that fronts open-weight models (Kimi K2.x, GLM-5.x, DeepSeek V4,
    MiniMax M2.x, MiMo V2.x, Qwen 3.x). No Claude or GPT here.
    Default model is `kimi-k2.6` because it's the strongest general
    open-weight model on the plan at the time of writing.
    """
    from engram.providers.openai import OpenAIChat

    return OpenAIChat(
        model=model or "kimi-k2.6",
        api_key=_opencode_api_key(),
        base_url="https://opencode.ai/zen/go/v1",
    )


def _openai_chat(model: str | None) -> ChatProvider:
    from engram.providers.openai import OpenAIChat

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAIChat(model=model or "gpt-4o-mini", api_key=api_key)


def _openai_embedder(model: str | None, dim: int | None) -> EmbeddingProvider:
    from engram.providers.openai import OpenAIEmbedder

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; the OpenAI embedder is required for real-provider runs."
        )
    return OpenAIEmbedder(
        model=model or "text-embedding-3-small",
        dim=dim if dim is not None else 1536,
        api_key=api_key,
    )


def _anthropic_chat(model: str | None) -> ChatProvider:
    from engram.providers.anthropic import AnthropicChat

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    return AnthropicChat(model=model or "claude-haiku-4-5-20251001", api_key=api_key)


_CHAT_BUILDERS: dict[str, Any] = {
    "fake": lambda model: FakeChat(),  # noqa: ARG005
    "openai": _openai_chat,
    "anthropic": _anthropic_chat,
    "moonshot": _moonshot_chat,
    "opencode-zen": _opencode_zen_chat,
    "opencode-go": _opencode_go_chat,
}

_EMBEDDER_BUILDERS: dict[str, Any] = {
    "fake": lambda model, dim: FakeEmbedder(dim=dim if dim is not None else 128),  # noqa: ARG005
    "openai": _openai_embedder,
}


def build_provider(
    *,
    embedder_name: str = "fake",
    chat_name: str = "fake",
    embed_model: str | None = None,
    embed_dim: int | None = None,
    chat_model: str | None = None,
) -> _MixedProvider:
    """Construct a bench Provider from CLI flags.

    Defaults are `fake/fake` so the existing CI smoke benchmark keeps
    working unchanged. Specify `embedder_name=openai` and `chat_name=
    openai|anthropic|moonshot` for real runs; missing API keys surface
    as actionable RuntimeError messages.
    """
    if embedder_name not in _EMBEDDER_BUILDERS:
        raise ValueError(
            f"unknown embedder {embedder_name!r}; choose from {sorted(_EMBEDDER_BUILDERS)}"
        )
    if chat_name not in _CHAT_BUILDERS:
        raise ValueError(f"unknown chat {chat_name!r}; choose from {sorted(_CHAT_BUILDERS)}")

    embedder = _EMBEDDER_BUILDERS[embedder_name](embed_model, embed_dim)
    chat = _CHAT_BUILDERS[chat_name](chat_model)
    name = f"{embedder_name}+{chat_name}"
    return _MixedProvider(name=name, embedder=embedder, chat=chat)
