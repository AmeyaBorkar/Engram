"""Format Memory retrieval results into prompt-ready context strings.

The cheapest, most framework-agnostic integration point. Callers
sprinkle this into any agent loop:

    context = format_context(memory.retrieve("user question", k=5))
    prompt = f"Relevant memories:\\n{context}\\n\\nUser: {question}"

No dependency on LangGraph / LlamaIndex / etc -- pure string
formatting over `RetrievalResult` and `ProcedureMatch`.
"""

from __future__ import annotations

from collections.abc import Sequence

from engram.schemas import ProcedureMatch, RetrievalResult


def format_context(
    results: Sequence[RetrievalResult | ProcedureMatch],
    *,
    include_score: bool = False,
    include_level: bool = True,
    bullet: str = "- ",
    max_items: int | None = None,
) -> str:
    """Format retrieval / procedure results as a multi-line bulleted string.

    Empty input returns the empty string. Each item becomes one line of
    the form `{bullet}[{level}, score={score:.2f}] {content}` (with
    optional score / level prefix).

    For procedure matches the line is `{bullet}[procedure] situation
    -> action` so the LLM sees the conditional structure.

    Args:
      results: a list of `RetrievalResult` or `ProcedureMatch`.
        Sequences are detected by element type (mixed lists are not
        supported -- pass each kind separately).
      include_score: prefix the score for tie-break visibility.
      include_level: prefix the level tag (`event`, `summary`, etc).
      bullet: line prefix; default `"- "` is markdown-friendly.
      max_items: cap; None = include all.
    """
    if not results:
        return ""
    items: Sequence[RetrievalResult | ProcedureMatch] = (
        list(results)[:max_items] if max_items is not None else list(results)
    )

    lines: list[str] = []
    for r in items:
        prefix_parts: list[str] = []
        if isinstance(r, ProcedureMatch):
            prefix_parts.append("procedure")
            if include_score:
                prefix_parts.append(f"score={r.score:.2f}")
            prefix = "[" + ", ".join(prefix_parts) + "]"
            lines.append(
                f"{bullet}{prefix} {r.procedure.situation} -> {r.procedure.action}"
            )
        else:
            if include_level:
                prefix_parts.append(r.level.value)
            if include_score:
                prefix_parts.append(f"score={r.score:.2f}")
            prefix = "[" + ", ".join(prefix_parts) + "] " if prefix_parts else ""
            lines.append(f"{bullet}{prefix}{r.content}")
    return "\n".join(lines)
