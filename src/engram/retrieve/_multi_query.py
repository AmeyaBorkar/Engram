"""Multi-query expansion + Reciprocal Rank Fusion.

A single query is one path through the embedding space. Two
paraphrased queries probe slightly different regions; their UNION
is more recall-complete than either alone. Reciprocal Rank Fusion
[Cormack, Clarke, Buettcher 2009] aggregates the per-query ranked
lists into a single fused ranking with no parameter to tune beyond
a single smoothing constant `k` (de-facto `k=60` from the RRF paper).

This module ships:

  * `expand_queries(query, n, chat)` -- chat-based paraphrase
    generation; returns `[query, paraphrase_1, ..., paraphrase_{n-1}]`
    so the original query is always represented.
  * `reciprocal_rank_fusion(per_query_rankings, k=60)` -- pure-math
    RRF over `list[list[RetrievalResult]]`. Returns one merged
    list, ordered by RRF score descending.

Memory.retrieve threads them together when `RetrieveParams.multi_query_n
>= 2` AND a chat provider is configured.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from importlib import resources

from engram.providers._message import Message
from engram.providers._protocols import ChatProvider
from engram.schemas import RetrievalResult
from engram._prompt_util import inline as _inline, render_prompt

MULTI_QUERY_PROMPT_NAME = "multi_query"
MULTI_QUERY_PROMPT_VERSION = "v1"
MULTI_QUERY_PROMPT_FILENAME = (
    f"{MULTI_QUERY_PROMPT_NAME}_{MULTI_QUERY_PROMPT_VERSION}.txt"
)


def load_multi_query_prompt() -> str:
    pkg = resources.files("engram.retrieve.prompts")
    return (pkg / MULTI_QUERY_PROMPT_FILENAME).read_text(encoding="utf-8")


def render_multi_query_prompt(query: str, n: int) -> str:
    template = load_multi_query_prompt()
    return render_prompt(template, query=_inline(query), n=str(n))


def expand_queries(query: str, n: int, chat: ChatProvider) -> list[str]:
    """Return `[query, p1, p2, ..., p_{n-1}]` -- the original plus
    `n-1` paraphrases via chat. Best-effort: on chat error or empty
    response, returns `[query]` so the caller can still retrieve."""
    if n <= 1:
        return [query]
    paraphrase_count = n - 1
    prompt = render_multi_query_prompt(query, paraphrase_count)
    try:
        response = chat.chat([Message(role="user", content=prompt)])
    except Exception:
        return [query]
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    # Drop blanks / leading numbering ("1." / "1)" / "- ").
    cleaned: list[str] = []
    for line in lines:
        stripped = _strip_leading_marker(line)
        if stripped and stripped != query:
            cleaned.append(stripped)
    # Always lead with the original query.
    return [query, *cleaned[:paraphrase_count]]


def reciprocal_rank_fusion(
    per_query_rankings: Sequence[Sequence[RetrievalResult]],
    *,
    k: int = 60,
) -> list[RetrievalResult]:
    """Fuse ranked lists via RRF.

    For each item d, the fused score is `sum_{q} 1 / (k + rank_q(d))`
    where `rank_q(d)` is 1-indexed; items missing from a list don't
    contribute. The smoothing `k=60` is from the RRF paper and
    rarely needs tuning.

    The returned `RetrievalResult` reuses the first occurrence of
    each item (preserving `level`, `content`, `supported_by`). Its
    `score` field is overwritten with the fused RRF score so the
    caller can re-rank trivially.
    """
    fused: dict[bytes, float] = defaultdict(float)
    first_occurrence: dict[bytes, RetrievalResult] = {}
    for ranking in per_query_rankings:
        for rank_idx, result in enumerate(ranking, start=1):
            key = result.item_id.bytes
            fused[key] += 1.0 / (k + rank_idx)
            if key not in first_occurrence:
                first_occurrence[key] = result
    # Re-emit sorted by fused score desc; replace .score with RRF.
    out = []
    for key, fused_score in fused.items():
        r = first_occurrence[key]
        out.append(
            RetrievalResult(
                item_id=r.item_id,
                level=r.level,
                content=r.content,
                confidence=r.confidence,
                score=fused_score,
                supported_by=r.supported_by,
            )
        )
    out.sort(key=lambda r: r.score, reverse=True)
    return out

def _strip_leading_marker(line: str) -> str:
    """Strip a leading list marker like '1. ' or '- '."""
    stripped = line.lstrip()
    if not stripped:
        return ""
    # Numbered: "1." / "1)" / "1: ".  We previously also matched a bare
    # space ("1 something") which corrupted real-content queries like
    # '5 wonderful things' into 'wonderful things'.  Require an
    # explicit punctuation separator so a leading digit that's part of
    # the query survives.
    if stripped[0].isdigit():
        for sep in (". ", ") ", ": "):
            idx = stripped.find(sep)
            if 0 < idx <= 3 and stripped[:idx].isdigit():
                return stripped[idx + len(sep) :].strip()
    # Bullet: "- " / "* " / "• "
    if stripped[0] in "-*•" and len(stripped) > 1 and stripped[1] == " ":
        return stripped[2:].strip()
    return stripped


__all__ = [
    "expand_queries",
    "load_multi_query_prompt",
    "reciprocal_rank_fusion",
    "render_multi_query_prompt",
]
