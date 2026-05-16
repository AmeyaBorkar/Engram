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

from importlib import resources

from engram.providers._message import Message
from engram.providers._protocols import ChatProvider

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


# The decomposition prompt instructs the LLM to emit between 2 and 5
# sub-questions.  An unbounded list lets a misbehaving model fan a
# single retrieve into N parallel leaf retrieves — each one a full
# embed + ANN + rerank pass.  Cap at 5 (prompt's documented max) to
# match the prompt's contract and bound the fan-out cost.
_MAX_DECOMPOSE_SUBQUERIES: int = 5


def decompose_query(query: str, chat: ChatProvider) -> list[str]:
    """Return a list of sub-queries via chat. Fails open: chat error
    or unparseable response returns `[query]` so the caller can still
    retrieve.

    The returned list always starts with the original query so the
    decomposed retrieve never gives up signal from the literal user
    phrasing.  Sub-queries are capped at 5 (the prompt's documented
    upper bound) so a misbehaving model can't drive an unbounded
    parallel fan-out.
    """
    prompt = render_decompose_prompt(query)
    try:
        response = chat.chat([Message(role="user", content=prompt)])
    except Exception:
        return [query]
    lines = [_strip_marker(line.strip()) for line in response.splitlines()]
    cleaned = [line for line in lines if line and line != query]
    cleaned = cleaned[:_MAX_DECOMPOSE_SUBQUERIES]
    return [query, *cleaned]


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
    # See consolidation/_abstraction._inline.
    return (
        content.replace("\\", "\\\\")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
        .replace(" ", "\\n")
        .replace(" ", "\\n")
        .replace("\t", "\\t")
    )


__all__ = [
    "decompose_query",
    "load_decompose_prompt",
    "render_decompose_prompt",
]
