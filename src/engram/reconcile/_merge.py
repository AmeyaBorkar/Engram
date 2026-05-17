"""LLM-driven merge for Stage 8's `Resolution.MERGE` policy.

The reconciler asks the chat provider to synthesize a single statement
that captures the truth behind two contradicting memory items, then
plants a new `MemoryItem` with that content + provenance union of both
parents. Both originals get `invalidated_by = merged_item.id`.

The prompt template (`prompts/merge_v1.txt`) is treated the same way
as the Stage 5 abstraction/judge prompts: payloads are inlined to
prevent newline-driven injection, the OUTPUT FORMAT block forces
JSON-only output, and parse failures fall back to a safe default
(here: the newer statement verbatim, mirroring the prompt's "if
irreconcilable" guidance).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources

from pydantic import BaseModel, ConfigDict, ValidationError

from engram.consolidation._abstraction import AbstractionParseError
from engram.providers._message import Message
from engram.providers._protocols import ChatProvider

MERGE_PROMPT_NAME = "merge"
MERGE_PROMPT_VERSION = "v1"
MERGE_PROMPT_FILENAME = f"{MERGE_PROMPT_NAME}_{MERGE_PROMPT_VERSION}.txt"


class MergeResponse(BaseModel):
    """Validated output of one merge call."""

    model_config = ConfigDict(frozen=True)

    merged: str


def load_merge_prompt() -> str:
    pkg = resources.files("engram.reconcile.prompts")
    return (pkg / MERGE_PROMPT_FILENAME).read_text(encoding="utf-8")


def render_merge_prompt(*, a: str, b: str) -> str:
    template = load_merge_prompt()
    return template.replace("{a}", _inline(a)).replace("{b}", _inline(b))


def parse_merge_response(text: str) -> MergeResponse:
    payload = _strip_code_fence(text).strip()
    if not payload:
        raise AbstractionParseError("empty merge response")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AbstractionParseError(f"invalid merge JSON: {exc.msg}") from exc
    try:
        return MergeResponse.model_validate(data)
    except ValidationError as exc:
        raise AbstractionParseError(f"merge schema mismatch: {exc}") from exc


@dataclass(frozen=True, slots=True)
class MergeOutcome:
    """Result of one `merge_with_status` call.

    `is_fallback` is True iff every retry's response failed parsing
    and the function returned the configured fallback (or `b`) verbatim.
    The reconciler pins this into the merged item's metadata
    (`reconcile.merge_fallback`) so operators can audit which merged
    rows came from a working judge call versus a defensive default.
    """

    content: str
    is_fallback: bool


def merge_with_status(
    *,
    a: str,
    b: str,
    chat: ChatProvider,
    max_retries: int = 1,
    fallback: str | None = None,
) -> MergeOutcome:
    """Run the LLM merge and signal whether the result is the fallback.

    Same fallback semantics as `merge()` (returns `fallback` if given,
    else `b`), but also exposes an `is_fallback` flag so callers can
    distinguish a synthesized merge from a defensive default.
    """
    prompt = render_merge_prompt(a=a, b=b)
    messages: list[Message] = [Message(role="user", content=prompt)]
    last_response = ""
    for _ in range(max_retries + 1):
        last_response = chat.chat(messages)
        try:
            return MergeOutcome(
                content=parse_merge_response(last_response).merged.strip(),
                is_fallback=False,
            )
        except AbstractionParseError:
            messages = [
                *messages,
                Message(role="assistant", content=last_response),
                Message(
                    role="user",
                    content=(
                        "Your previous response was not valid JSON matching the schema. "
                        "Respond only with the JSON object, no surrounding prose."
                    ),
                ),
            ]
    return MergeOutcome(
        content=fallback if fallback is not None else b,
        is_fallback=True,
    )


def merge(
    *,
    a: str,
    b: str,
    chat: ChatProvider,
    max_retries: int = 1,
    fallback: str | None = None,
) -> str:
    """Run the LLM merge on a pair of contradicting statements.

    On any parse failure, returns `fallback` if given, else `b` (the
    newer statement) -- mirroring the prompt's "if irreconcilable,
    output B" guidance. Stage 8's reconciler always passes `b` as the
    newer-created item so the fallback is conservatively sane.

    Thin wrapper over `merge_with_status` that drops the fallback
    signal; preserved for callers (none in the engine after M-194)
    that only need the content string.
    """
    return merge_with_status(
        a=a, b=b, chat=chat, max_retries=max_retries, fallback=fallback
    ).content


_FENCE_RE = re.compile(
    r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$",
    flags=re.DOTALL,
)


def _strip_code_fence(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _inline(content: str) -> str:
    return content.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


__all__ = [
    "MERGE_PROMPT_VERSION",
    "MergeOutcome",
    "MergeResponse",
    "load_merge_prompt",
    "merge",
    "merge_with_status",
    "parse_merge_response",
    "render_merge_prompt",
]
