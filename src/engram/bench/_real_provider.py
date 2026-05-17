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


def _moonshot_chat(model: str | None, max_tokens: int | None = None) -> ChatProvider:
    from engram.providers.openai import OpenAIChat

    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOONSHOT_API_KEY is not set. Get a key from https://platform.moonshot.ai/"
        )
    kwargs: dict[str, Any] = {
        "model": model or "kimi-k2.6",
        "api_key": api_key,
        "base_url": "https://api.moonshot.ai/v1",
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return OpenAIChat(**kwargs)


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


def _opencode_zen_chat(model: str | None, max_tokens: int | None = None) -> ChatProvider:
    """OpenCode Zen — multi-model gateway with OpenAI-compatible chat API.

    One API key, access to Claude (haiku/sonnet/opus 4.x), GPT 5.x, and
    Kimi K2.x. No embedding model -- pair with `--embedder openai`.
    Default model is `claude-haiku-4-5` because it's cheap, fast, and
    matches what most LongMemEval-style reports use.
    """
    from engram.providers.openai import OpenAIChat

    kwargs: dict[str, Any] = {
        "model": model or "claude-haiku-4-5",
        "api_key": _opencode_api_key(),
        "base_url": "https://opencode.ai/zen/v1",
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return OpenAIChat(**kwargs)


def _opencode_go_chat(model: str | None, max_tokens: int | None = None) -> ChatProvider:
    """OpenCode Go -- open-weight coding models behind one subscription.

    Same OpenCode account / API key as Zen, but a different endpoint
    that fronts open-weight models (Kimi K2.x, GLM-5.x, DeepSeek V4,
    MiniMax M2.x, MiMo V2.x, Qwen 3.x). No Claude or GPT here.
    Default model is `kimi-k2.6` because it's the strongest general
    open-weight model on the plan at the time of writing.

    `max_tokens` defaults to 8192 because Kimi K2.6 (and other thinking-
    capable open-weight models on this endpoint) reason for 1000-3000
    tokens before emitting the final answer. The OpenAIChat default of
    1024 is a safety guard for unknown endpoints, but here it would cut
    the model off mid-reasoning before any answer is emitted -- see
    JOURNEY §24 for the diagnostic trail. An explicit `max_tokens` arg
    (e.g. from `--chat-max-tokens`) overrides this default.
    """
    from engram.providers.openai import OpenAIChat

    return OpenAIChat(
        model=model or "kimi-k2.6",
        api_key=_opencode_api_key(),
        base_url="https://opencode.ai/zen/go/v1",
        max_tokens=max_tokens if max_tokens is not None else 8192,
    )


def _openai_chat(model: str | None, max_tokens: int | None = None) -> ChatProvider:
    from engram.providers.openai import OpenAIChat

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    kwargs: dict[str, Any] = {"model": model or "gpt-4o-mini", "api_key": api_key}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return OpenAIChat(**kwargs)


def _openrouter_headers() -> dict[str, str]:
    """Optional ranking headers OpenRouter uses for its public leaderboards.

    Both headers are optional but encouraged -- they let OpenRouter
    attribute the call to the originating app. Overridable via the
    standard env vars; default to Engram so SOTA-bench runs show up
    correctly on the leaderboard.
    """
    return {
        "HTTP-Referer": os.environ.get(
            "OPENROUTER_HTTP_REFERER", "https://github.com/AmeyaBorkar/Engram"
        ),
        "X-Title": os.environ.get("OPENROUTER_X_TITLE", "Engram"),
    }


def _openrouter_chat(model: str | None, max_tokens: int | None = None) -> ChatProvider:
    """OpenRouter chat (OpenAI-compatible) -- one API key, every frontier model.

    Catalog includes `anthropic/claude-opus-4.7`, `openai/gpt-5.5`,
    `moonshotai/kimi-k2.6`, `deepseek/deepseek-v4-pro`,
    `google/gemini-3-pro`, `x-ai/grok-4.3`, etc. Default is
    `anthropic/claude-haiku-4-5` because it matches what most
    LongMemEval-style reports use and stays cheap. Override via
    --chat-model.

    `max_tokens` is left unset by default (falls through to OpenAIChat's
    1024 safety cap). Thinking-mode models routed through OpenRouter
    (e.g. `moonshotai/kimi-k2.6`) need an explicit `--chat-max-tokens
    8192` to avoid the mid-reasoning truncation documented in JOURNEY
    §24 -- the cap that bit us on the opencode-go endpoint applies
    identically here because the underlying model is the same.
    """
    from engram.providers.openai import OpenAIChat

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Get a key from https://openrouter.ai/keys "
            "and add it to your .env."
        )
    kwargs: dict[str, Any] = {
        "model": model or "anthropic/claude-haiku-4-5",
        "api_key": api_key,
        "base_url": "https://openrouter.ai/api/v1",
        "default_headers": _openrouter_headers(),
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return OpenAIChat(**kwargs)


def _openrouter_embedder(
    model: str | None,
    dim: int | None,
    device: str | None = None,  # noqa: ARG001 - unused; uniform builder shape
) -> EmbeddingProvider:
    """OpenRouter embeddings -- one API key for both chat AND embeddings.

    Default `qwen/qwen3-embedding-8b` (MTEB ~70.6, $0.01/M tokens,
    4096-dim, normalized) is the strongest single-key option on
    OpenRouter for benchmark runs. Other strong picks:
      * `google/gemini-embedding-001` (3072 dim, multilingual)
      * `openai/text-embedding-3-large` (3072 dim, supports
        `dimensions=` truncation)
      * `baai/bge-m3` (1024 dim, multilingual, cheap)

    `dim` defaults to the model's native size; pass an explicit
    `--embed-dim` only if you want truncation AND the chosen model
    supports it (currently only the `openai/*` routes).
    """
    from engram.providers.openai import OpenAIEmbedder

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Get a key from https://openrouter.ai/keys "
            "and add it to your .env."
        )
    chosen_model = model or "qwen/qwen3-embedding-8b"
    # Auto-fill dim from the catalog so callers don't have to know it.
    if dim is None:
        from engram.providers.openai import _OPENAI_NATIVE_EMBED_DIMS

        catalog_dim = _OPENAI_NATIVE_EMBED_DIMS.get(chosen_model, -1)
        if catalog_dim < 0:
            raise RuntimeError(
                f"openrouter embedder {chosen_model!r} has an unknown native dim; "
                f"pass --embed-dim N to set it explicitly."
            )
        dim = catalog_dim
    return OpenAIEmbedder(
        model=chosen_model,
        dim=dim,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers=_openrouter_headers(),
    )


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


def _local_embedder(
    model: str | None,
    dim: int | None,
    device: str | None = None,
    dtype: str = "auto",
) -> EmbeddingProvider:
    """sentence-transformers behind a GPU when available, CPU otherwise.

    Default `BAAI/bge-large-en-v1.5` ships excellent retrieval quality
    without an API key. The model downloads once into the HuggingFace
    cache; subsequent runs are warm. Pass `device="cpu"` to force CPU
    when CUDA is detected but broken (driver/architecture mismatch).
    `dtype="auto"` (default) -> fp16 on CUDA, fp32 elsewhere.
    """
    from engram.providers.local import LocalEmbedder

    return LocalEmbedder(
        model=model or "BAAI/bge-large-en-v1.5",
        dim=dim,
        device=device,
        dtype=dtype,  # type: ignore[arg-type]
    )


def _anthropic_chat(model: str | None, max_tokens: int | None = None) -> ChatProvider:
    from engram.providers.anthropic import AnthropicChat

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    kwargs: dict[str, Any] = {
        "model": model or "claude-haiku-4-5-20251001",
        "api_key": api_key,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return AnthropicChat(**kwargs)


_CHAT_BUILDERS: dict[str, Any] = {
    "fake": lambda model, max_tokens=None: FakeChat(),  # noqa: ARG005
    "openai": _openai_chat,
    "anthropic": _anthropic_chat,
    "moonshot": _moonshot_chat,
    "opencode-zen": _opencode_zen_chat,
    "opencode-go": _opencode_go_chat,
    "openrouter": _openrouter_chat,
}

_EMBEDDER_BUILDERS: dict[str, Any] = {
    # `_*` builders take (model, dim, device, dtype). The lambdas below
    # ignore what they don't need so the call signature stays uniform.
    "fake": lambda model, dim, device, dtype: FakeEmbedder(  # noqa: ARG005
        dim=dim if dim is not None else 128
    ),
    "openai": lambda model, dim, device, dtype: _openai_embedder(  # noqa: ARG005
        model, dim
    ),
    "local": _local_embedder,
    "openrouter": lambda model, dim, device, dtype: _openrouter_embedder(  # noqa: ARG005
        model, dim, device
    ),
}


def build_chat(
    name: str,
    model: str | None = None,
    *,
    max_tokens: int | None = None,
) -> ChatProvider:
    """Construct a standalone chat provider by name.

    Same chat catalog as `build_provider`, but returns the chat
    provider alone -- useful for the bench's secondary chat slots
    (`consolidate_chat`, `judge_chat`) where the embedder side is
    irrelevant.

    `max_tokens`, when set, overrides the per-builder default. `None`
    (the default) preserves backwards compat -- each builder applies
    whatever cap it chose (opencode-go's 8192 for thinking-mode Kimi,
    others fall through to OpenAIChat's 1024 safety guard).
    """
    if name not in _CHAT_BUILDERS:
        raise ValueError(f"unknown chat {name!r}; choose from {sorted(_CHAT_BUILDERS)}")
    chat: ChatProvider = _CHAT_BUILDERS[name](model, max_tokens)
    return chat


def build_provider(
    *,
    embedder_name: str = "fake",
    chat_name: str = "fake",
    embed_model: str | None = None,
    embed_dim: int | None = None,
    embed_device: str | None = None,
    embed_dtype: str = "auto",
    chat_model: str | None = None,
    chat_max_tokens: int | None = None,
) -> _MixedProvider:
    """Construct a bench Provider from CLI flags.

    Defaults are `fake/fake` so the existing CI smoke benchmark keeps
    working unchanged. Specify `embedder_name=openai|local|openrouter`
    and `chat_name=openai|anthropic|moonshot|opencode-zen|opencode-go|openrouter`
    for real runs; missing API keys surface as actionable RuntimeError
    messages. `embed_device` + `embed_dtype` only apply to local.

    `chat_max_tokens`, when set, overrides the per-builder default cap.
    Required when routing a thinking-mode model (e.g. Kimi K2.6) through
    a generic OpenAI-compatible endpoint like OpenRouter that otherwise
    inherits OpenAIChat's 1024-token safety guard -- see JOURNEY §24.
    """
    if embedder_name not in _EMBEDDER_BUILDERS:
        raise ValueError(
            f"unknown embedder {embedder_name!r}; choose from {sorted(_EMBEDDER_BUILDERS)}"
        )
    if chat_name not in _CHAT_BUILDERS:
        raise ValueError(f"unknown chat {chat_name!r}; choose from {sorted(_CHAT_BUILDERS)}")

    embedder = _EMBEDDER_BUILDERS[embedder_name](
        embed_model, embed_dim, embed_device, embed_dtype
    )
    chat = _CHAT_BUILDERS[chat_name](chat_model, chat_max_tokens)
    # M-151: name includes the resolved model so a sweep over
    # `--chat-model` produces distinguishable rows in the
    # SCOREBOARD's first column. Pre-fix the name collapsed every
    # OpenAI run to `openai+openai` regardless of which model was
    # behind it; the manifest_hash distinguished them, but readers
    # had to cross-reference SCOREBOARD prose to know.
    embedder_model = getattr(embedder, "model", "?")
    chat_model_label = getattr(chat, "model", "?")
    name = f"{embedder_name}:{embedder_model}+{chat_name}:{chat_model_label}"
    return _MixedProvider(name=name, embedder=embedder, chat=chat)
