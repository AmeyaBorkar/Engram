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
import logging
from dataclasses import dataclass
from importlib import resources

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from engram._prompt_util import (
    inline as _inline,
)
from engram._prompt_util import (
    render_prompt,
)
from engram._prompt_util import (
    strip_code_fence as _strip_code_fence,
)
from engram.consolidation._abstraction import AbstractionParseError
from engram.providers._message import Message
from engram.providers._protocols import ChatProvider

MERGE_PROMPT_NAME = "merge"
MERGE_PROMPT_VERSION = "v1"
MERGE_PROMPT_FILENAME = f"{MERGE_PROMPT_NAME}_{MERGE_PROMPT_VERSION}.txt"

_LOG = logging.getLogger("engram.reconcile.merge")


class MergeResponse(BaseModel):
    """Validated output of one merge call.

    Bounded match for `MemoryItem.content`: same upper cap (64 KiB)
    and a non-empty/non-whitespace requirement so a model that returns
    `{"merged": ""}` or `{"merged": " "}` can't plant an empty memory
    item that downstream consumers would treat as 'real'.
    """

    model_config = ConfigDict(frozen=True)

    merged: str = Field(min_length=1, max_length=64 * 1024)

    @model_validator(mode="after")
    def _check_non_whitespace(self) -> MergeResponse:
        if not self.merged.strip():
            raise ValueError("merged content must not be whitespace-only")
        return self


@dataclass(frozen=True, slots=True)
class MergeOutcome:
    """Tagged result of `merge_with_status`.

    `is_fallback=False` means the LLM produced a valid `MergeResponse`
    and we are returning its synthesized content. `is_fallback=True`
    means the LLM exhausted its retries without producing parseable
    output and the caller is being handed the conservative fallback
    text (typically the newer statement verbatim). Callers that plant
    a new memory item with the merged content should pin a
    `metadata["reconcile"]["merge_fallback"] = True` flag on it so
    operators can audit synthesized-vs-laundered merges (audit H-05,
    M-194).
    """

    merged: str
    is_fallback: bool


def load_merge_prompt() -> str:
    pkg = resources.files("engram.reconcile.prompts")
    return (pkg / MERGE_PROMPT_FILENAME).read_text(encoding="utf-8")


def render_merge_prompt(*, a: str, b: str) -> str:
    template = load_merge_prompt()
    # Single-pass substitution: a chained .replace("{a}",a).replace("{b}",b)
    # lets `a` contain the literal text `{b}` and steer the second replace
    # into the wrong slot (audit H-03).  render_prompt walks the template
    # once and substitutes each placeholder at most once.
    return render_prompt(template, a=_inline(a), b=_inline(b))


def parse_merge_response(text: str) -> MergeResponse:
    # Local import keeps the security check optional from a packaging
    # standpoint and mirrors `_abstraction.parse_response` (audit H-04).
    from engram._security.prompt_injection import looks_like_injection

    payload = _strip_code_fence(text).strip()
    if not payload:
        raise AbstractionParseError("empty merge response")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AbstractionParseError(f"invalid merge JSON: {exc.msg}") from exc
    try:
        response = MergeResponse.model_validate(data)
    except ValidationError as exc:
        raise AbstractionParseError(f"merge schema mismatch: {exc}") from exc
    # Audit H-04: the abstraction path rejects injection-laden output;
    # merge had no equivalent gate. Without this an LLM that echoes
    # attacker text into `merged` plants the injection as a fresh
    # MemoryItem at the merged tier. Triggering this raises
    # AbstractionParseError so the retry-or-fallback loop runs.
    if looks_like_injection(response.merged):
        raise AbstractionParseError(
            "merged content contains prompt-injection-like content; rejecting"
        )
    return response


def merge_with_status(
    *,
    a: str,
    b: str,
    chat: ChatProvider,
    max_retries: int = 1,
    fallback: str | None = None,
) -> MergeOutcome:
    """Run the LLM merge and return both the content and a fallback flag.

    On success returns `MergeOutcome(merged=<llm output>,
    is_fallback=False)`. When every retry fails to produce a parseable
    `MergeResponse`, returns the conservative fallback in a tagged
    `MergeOutcome(is_fallback=True)` so the caller can pin
    audit metadata (`merge_fallback: True`) on the planted item and
    refuse to plant if the fallback text itself trips the injection
    screen (audit H-05).

    If the fallback text (caller-supplied or the default `b`) looks
    like prompt injection, raises `AbstractionParseError` rather than
    laundering the attacker payload. The caller is expected to handle
    this by skipping the merge entirely (e.g. fall back to a different
    `Resolution`).
    """
    from engram._security.prompt_injection import looks_like_injection

    prompt = render_merge_prompt(a=a, b=b)
    messages: list[Message] = [Message(role="user", content=prompt)]
    last_response = ""
    for _ in range(max_retries + 1):
        last_response = chat.chat(messages)
        try:
            return MergeOutcome(
                merged=parse_merge_response(last_response).merged.strip(),
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
    fallback_text = fallback if fallback is not None else b
    # Audit H-05: if the fallback text itself trips the injection screen
    # we MUST NOT launder it into a fresh memory item. Raise so the
    # caller can pick a non-merging resolution.
    if looks_like_injection(fallback_text):
        raise AbstractionParseError(
            "merge fallback text contains prompt-injection-like content; refusing to plant"
        )
    _LOG.warning(
        "reconcile.merge: LLM produced no parseable response after %d attempt(s); "
        "returning fallback content (length=%d)",
        max_retries + 1,
        len(fallback_text),
    )
    return MergeOutcome(merged=fallback_text, is_fallback=True)


def merge(
    *,
    a: str,
    b: str,
    chat: ChatProvider,
    max_retries: int = 1,
    fallback: str | None = None,
) -> str:
    """Run the LLM merge on a pair of contradicting statements.

    Back-compat shim returning only the merged string. New code should
    prefer `merge_with_status` so it can distinguish a synthesized
    merge from a fallback and pin audit metadata accordingly (audit
    H-05, M-194).

    On any parse failure, returns `fallback` if given, else `b` (the
    newer statement) -- mirroring the prompt's "if irreconcilable,
    output B" guidance. Stage 8's reconciler always passes `b` as the
    newer-created item so the fallback is conservatively sane.
    """
    return merge_with_status(a=a, b=b, chat=chat, max_retries=max_retries, fallback=fallback).merged


__all__ = [
    "MERGE_PROMPT_FILENAME",
    "MERGE_PROMPT_NAME",
    "MERGE_PROMPT_VERSION",
    "MergeOutcome",
    "MergeResponse",
    "load_merge_prompt",
    "merge",
    "merge_with_status",
    "parse_merge_response",
    "render_merge_prompt",
]
