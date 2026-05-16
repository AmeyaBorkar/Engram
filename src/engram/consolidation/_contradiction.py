"""Contradiction detection for newly consolidated abstractions.

When the engine produces a new abstraction it can shadow or conflict
with an existing one. Stage 5's job is to *detect* the conflict; Stage 8
ships the resolution policy. The detector is a two-stage filter:

  1. Vector recall: find existing memory items whose embedding is close
     to the new abstraction's embedding (cosine similarity above
     `similarity_threshold`). This is the cheap pass - one indexed read,
     no LLM calls.
  2. LLM judge: for each candidate above the threshold, ask the chat
     provider to classify the relationship as "agree", "contradict",
     or "unrelated". Contradictions are recorded; the rest are dropped.

Recording: the verdict goes into the NEW item's
`metadata["consolidation"]["conflicts"]` as a list of `{candidate_id,
similarity, verdict}` entries. Stage 8 walks both directions of the
graph (new <-> candidate) by scanning metadata.

The judge prompt is versioned (`judge_v1.txt`) and hardened against
prompt injection in the same way as `abstract_v1.txt`. Strict JSON
output, parsed with Pydantic, no free-form prose accepted.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError

from engram.consolidation._abstraction import AbstractionParseError
from engram.providers._message import Message
from engram.providers._protocols import ChatProvider
from engram.schemas import Verdict
from engram._security.prompt_injection import looks_like_injection
from engram._prompt_util import (
    inline as _inline,
    render_prompt,
    strip_code_fence as _strip_code_fence,
)

_LOG = logging.getLogger(__name__)

JUDGE_PROMPT_NAME = "judge"
JUDGE_PROMPT_VERSION = "v1"
JUDGE_PROMPT_FILENAME = f"{JUDGE_PROMPT_NAME}_{JUDGE_PROMPT_VERSION}.txt"


class JudgeResponse(BaseModel):
    """Validated output of one judge call."""

    model_config = ConfigDict(frozen=True)

    verdict: Verdict


@dataclass(frozen=True, slots=True)
class ContradictionParams:
    """Parameters of the contradiction-detection pass.

    `enabled=False` skips the LLM judge entirely - useful when the chat
    provider is metered and the abstraction quality is good enough that
    contradictions would be vanishingly rare. Off by default.

    `similarity_threshold` is the cosine-similarity floor for becoming a
    candidate; below this no judge call is made. Tuning matters: too
    low and we burn LLM calls on unrelated abstractions; too high and
    we miss real conflicts.

    `max_candidates` caps the per-call LLM cost.
    """

    enabled: bool = False
    similarity_threshold: float = 0.7
    max_candidates: int = 3
    max_retries: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {self.similarity_threshold!r}"
            )
        if self.max_candidates < 0:
            raise ValueError(f"max_candidates must be >= 0, got {self.max_candidates}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


@dataclass(frozen=True, slots=True)
class DetectedConflict:
    """One detector-output contradiction between a new abstraction and a candidate.

    Transient -- the engine collects a list of these per consolidation
    call and writes them to the new item's
    `metadata["consolidation"]["conflicts"]`. Stage 8's first-class
    `engram.schemas.Conflict` is the persistent storage entity that
    survives reconciliation; `DetectedConflict` is the in-flight
    detector record that gets persisted as a `Conflict` row.
    """

    candidate_id: UUID
    similarity: float
    verdict: Verdict


@dataclass(frozen=True, slots=True)
class CandidateRow:
    """One candidate the vector recall surfaced.

    The engine builds these from `Storage.search_memory_item_embeddings`
    output and feeds them to the judge. Carrying both `id` and `content`
    keeps the judge call self-contained (no second storage round-trip).
    """

    item_id: UUID
    content: str
    similarity: float


def load_judge_prompt() -> str:
    pkg = resources.files("engram.consolidation.prompts")
    return (pkg / JUDGE_PROMPT_FILENAME).read_text(encoding="utf-8")


def render_judge_prompt(*, a: str, b: str) -> str:
    template = load_judge_prompt()
    # Single-pass: chained .replace lets `a` contain a literal `{b}` and
    # redirect into the other slot (audit H-03).
    return render_prompt(template, a=_inline(a), b=_inline(b))


def parse_judge_response(text: str) -> JudgeResponse:
    payload = _strip_code_fence(text).strip()
    if not payload:
        raise AbstractionParseError("empty judge response")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AbstractionParseError(f"invalid judge JSON: {exc.msg}") from exc
    try:
        return JudgeResponse.model_validate(data)
    except ValidationError as exc:
        raise AbstractionParseError(f"judge schema mismatch: {exc}") from exc


def judge(
    *,
    a: str,
    b: str,
    chat: ChatProvider,
    max_retries: int = 1,
) -> Verdict:
    """Run the LLM judge on a pair of statements.

    On any parse failure, returns `Verdict.UNRELATED` (the safe default:
    we'd rather miss a real conflict than spuriously flag one). The
    Stage 8 resolver re-scans periodically anyway.

    The judge inputs are user-content; if either statement looks like
    a prompt-injection attempt we short-circuit to UNRELATED so the
    chat provider never sees the adversarial text.  Even though the
    judge's structured output limits the blast radius, sending an
    obviously-malicious payload through the LLM (a) bills tokens, and
    (b) gives the model a chance to leak the system prompt back
    through .reasoning fields if the schema is ever widened.
    """
    if looks_like_injection(a) or looks_like_injection(b):
        _LOG.debug("judge: short-circuit UNRELATED on injection-shaped input")
        return Verdict.UNRELATED
    prompt = render_judge_prompt(a=a, b=b)
    messages: list[Message] = [Message(role="user", content=prompt)]
    last_response = ""
    parse_failures = 0
    for _ in range(max_retries + 1):
        last_response = chat.chat(messages)
        try:
            return parse_judge_response(last_response).verdict
        except AbstractionParseError:
            parse_failures += 1
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
    # All retries exhausted with parse failures.  Log so a misbehaving
    # chat provider (consistently malformed JSON) surfaces to operators
    # rather than producing a silent stream of UNRELATED verdicts.
    _LOG.warning(
        "judge: parse failed across %d attempts, returning UNRELATED",
        parse_failures,
    )
    return Verdict.UNRELATED


def detect_contradictions(
    *,
    new_abstraction: str,
    candidates: list[CandidateRow],
    chat: ChatProvider,
    params: ContradictionParams,
) -> list[DetectedConflict]:
    """Run the judge against each candidate; collect the contradictions.

    The vector-recall step is the engine's responsibility (it has the
    storage handle); this function takes the already-filtered candidate
    list and only handles the LLM half. Returns conflicts in input
    order.
    """
    if not params.enabled or not candidates:
        return []
    out: list[DetectedConflict] = []
    for cand in candidates[: params.max_candidates]:
        verdict = judge(
            a=new_abstraction,
            b=cand.content,
            chat=chat,
            max_retries=params.max_retries,
        )
        if verdict is Verdict.CONTRADICT:
            out.append(
                DetectedConflict(
                    candidate_id=cand.item_id,
                    similarity=cand.similarity,
                    verdict=verdict,
                )
            )
    return out


def conflicts_to_metadata(conflicts: list[DetectedConflict]) -> list[dict[str, Any]]:
    """Stable JSON shape for `MemoryItem.metadata['consolidation']['conflicts']`."""
    return [
        {
            "candidate_id": str(c.candidate_id),
            "similarity": c.similarity,
            "verdict": c.verdict.value,
        }
        for c in conflicts
    ]

# Re-export for the engine.
__all__ = [
    "CandidateRow",
    "ContradictionParams",
    "DetectedConflict",
    "JudgeResponse",
    "Verdict",
    "conflicts_to_metadata",
    "detect_contradictions",
    "judge",
    "parse_judge_response",
    "render_judge_prompt",
]
