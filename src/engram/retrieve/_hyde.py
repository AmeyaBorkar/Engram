"""HyDE -- Hypothetical Document Embeddings for retrieval.

Standard retrieval pipelines embed the raw query, then look for the
closest memory by cosine similarity. The problem: questions and
answers are phrased differently. "What does the user prefer?" has
little textual overlap with "user prefers tabs over spaces" --
embedders only weakly bridge the gap.

HyDE [Gao et al. 2022, arXiv:2212.10496] sidesteps this by generating
a hypothetical *answer* via the chat provider and embedding that
instead. The hypothetical reads like a memory would; the embedder
brings it close to the real memory.

This module is the prompt+invoke wrapper. Memory.retrieve calls it
when `RetrieveParams.hyde=True` and a chat provider is configured.

Cost: one extra chat call per retrieve. The provider's `Cache`
wrapper deduplicates identical queries, so a follow-up retrieval on
the same query is free. The reranker pipeline downstream can still
re-rank against the original query so we don't lose intent signal.
"""

from __future__ import annotations

import logging
from importlib import resources

from engram._prompt_util import inline as _inline, render_prompt
from engram.providers._message import Message
from engram.providers._protocols import ChatProvider

_LOG = logging.getLogger("engram.retrieve")

HYDE_PROMPT_NAME = "hyde"
HYDE_PROMPT_VERSION = "v1"
HYDE_PROMPT_FILENAME = f"{HYDE_PROMPT_NAME}_{HYDE_PROMPT_VERSION}.txt"


def load_hyde_prompt() -> str:
    pkg = resources.files("engram.retrieve.prompts")
    return (pkg / HYDE_PROMPT_FILENAME).read_text(encoding="utf-8")


def render_hyde_prompt(query: str) -> str:
    template = load_hyde_prompt()
    return render_prompt(template, query=_inline(query))


def hyde_transform(query: str, chat: ChatProvider) -> str:
    """Return a hypothetical answer for `query` via the chat provider.

    On any chat error the original query is returned (HyDE is a best-
    effort precision boost, not a load-bearing primitive). The caller
    can also pass `hyde=False` to skip the LLM call entirely.
    """
    prompt = render_hyde_prompt(query)
    try:
        response = chat.chat([Message(role="user", content=prompt)])
    except Exception as exc:
        # Best-effort fallback: HyDE is a precision boost, not load-
        # bearing.  Log so a misconfigured chat provider surfaces to
        # operators as a warning instead of a silent feature-disable.
        _LOG.warning("hyde transform: chat raised %s: %s; falling back to raw query", type(exc).__name__, exc)
        return query
    cleaned = response.strip()
    if not cleaned:
        return query
    # If the hypothetical is shorter than the query, the chat probably
    # refused / returned filler -- fall back to the raw query.
    if len(cleaned) < max(8, len(query) // 4):
        return query
    return cleaned

__all__ = [
    "hyde_transform",
    "load_hyde_prompt",
    "render_hyde_prompt",
]
