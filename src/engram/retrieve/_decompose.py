"""Query decomposition for multi-hop / multi-fact retrieval.

A complex question often asks for several facts at once: "where does
the user live and what do they do for work?" Embedding this as one
query splits the budget across two unrelated regions of the
embedding space; retrieval picks up a bit of each and may miss the
strongest hit for either.

Decomposition asks the chat provider to break the query into
focused sub-queries. Each sub-query is retrieved independently; the
per-sub-query rankings are fused via the same Reciprocal Rank Fusion
helper as multi-query expansion.

Cost: one chat call to decompose, then N retrieves. Same shape as
multi-query expansion (E.3); the difference is intent: multi-query
paraphrases, decompose splits.
"""

from __future__ import annotations

import logging
from importlib import resources

from engram.providers._message import Message
from engram.providers._protocols import ChatProvider

_LOG = logging.getLogger("engram.retrieve")

# `decompose_query` returns at most this many sub-queries (excluding
# the original). Matches the contract in `prompts/decompose_v1.txt`.
# If the LLM emits more lines, we trim to the cap.
_DECOMPOSE_MAX_SUBQUERIES = 5

DECOMPOSE_PROMPT_NAME = "decompose"
DECOMPOSE_PROMPT_VERSION = "v1"
DECOMPOSE_PROMPT_FILENAME = (
    f"{DECOMPOSE_PROMPT_NAME}_{DECOMPOSE_PROMPT_VERSION}.txt"
)


def load_decompose_prompt() -> str:
    pkg = resources.files("engram.retrieve.prompts")
    return (pkg / DECOMPOSE_PROMPT_FILENAME).read_text(encoding="utf-8")


def render_decompose_prompt(query: str) -> str:
    template = load_decompose_prompt()
    return template.replace("{query}", _inline(query))


def decompose_query(query: str, chat: ChatProvider) -> list[str]:
    """Return a list of sub-queries via chat. Fails open: chat error
    or unparseable response returns `[query]` so the caller can still
    retrieve.

    The returned list always starts with the original query so the
    decomposed retrieve never gives up signal from the literal user
    phrasing. The list is capped at `1 + _DECOMPOSE_MAX_SUBQUERIES`
    elements -- the prompt contract is "emit at most 5 sub-queries"
    so any over-fill from a non-compliant LLM is trimmed here rather
    than leaking into N extra retrieves.
    """
    prompt = render_decompose_prompt(query)
    try:
        response = chat.chat([Message(role="user", content=prompt)])
    except Exception as exc:
        _LOG.warning(
            "decompose_query fell back to [query]: chat raised %s: %s",
            type(exc).__name__,
            exc,
            extra={"event": "engram.retrieve.decompose_failed"},
        )
        return [query]
    lines = [_strip_marker(line.strip()) for line in response.splitlines()]
    cleaned = [line for line in lines if line and line != query]
    return [query, *cleaned[:_DECOMPOSE_MAX_SUBQUERIES]]


def _strip_marker(line: str) -> str:
    """Strip a leading '1. ', '1) ', '- ', or '* ' marker."""
    if not line:
        return ""
    if line[0].isdigit():
        for sep in (". ", ") ", ": "):
            idx = line.find(sep)
            if 0 < idx <= 3 and line[:idx].isdigit():
                return line[idx + len(sep) :].strip()
    if line[0] in "-*•" and len(line) > 1 and line[1] == " ":
        return line[2:].strip()
    return line


def _inline(content: str) -> str:
    return content.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


__all__ = [
    "DECOMPOSE_PROMPT_VERSION",
    "decompose_query",
    "load_decompose_prompt",
    "render_decompose_prompt",
]
