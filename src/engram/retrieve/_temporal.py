"""Temporal anchor extraction (Tier 4 / E.13).

Temporal questions are the lowest-scoring category on LongMemEval for
basically every memory system. The standard failure mode: the LLM
can do date math but does it badly, and the retrieve path doesn't
get a temporal filter signal so the wrong-time answers surface.

This module bridges that gap. Given a question and the current time,
it asks the chat provider to emit a JSON anchor:

  { "anchor": "<ISO-8601 UTC, or null>", "reasoning": "..." }

If the question is non-temporal (no date reference at all), the LLM
returns `anchor=null`. Otherwise, the caller plugs the anchor into
`Memory.retrieve(..., as_of=anchor)`.

We use a JSON-anchor approach rather than LLM-generated Python code
(the TReMu approach) because:

  * JSON parsing has a predictable failure mode.
  * No sandbox-escape risk.
  * Modern LLMs (GPT-4 / Claude / Kimi K2.6) handle relative date
    math reliably when asked to compute the anchor directly.

A `compute_temporal_anchor(question, chat, now)` helper handles
everything; `Memory.retrieve(..., temporal=True)` wires it in.

Fails open: chat error, unparseable response, or unparseable
ISO-8601 -> anchor=None. The retrieve path then uses no temporal
filter, matching the default (current-state) behavior.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from importlib import resources

from pydantic import BaseModel, ConfigDict, ValidationError

from engram.consolidation._abstraction import AbstractionParseError
from engram.providers._message import Message
from engram.providers._protocols import ChatProvider

TEMPORAL_PROMPT_NAME = "temporal_anchor"
TEMPORAL_PROMPT_VERSION = "v1"
TEMPORAL_PROMPT_FILENAME = (
    f"{TEMPORAL_PROMPT_NAME}_{TEMPORAL_PROMPT_VERSION}.txt"
)


# Heuristic gate so we don't burn a chat call on every retrieve. If
# the question contains no temporal cue, skip the codegen prompt.
_NUM_WORDS = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|few|several|many|couple|dozen)"
)
_TIME_UNITS = r"(?:day|days|week|weeks|month|months|year|years|hour|hours)"

_TEMPORAL_CUES: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\b("
        rf"yesterday|today|tomorrow|tonight|last\s+(week|month|year|night|"
        rf"\w+day)|next\s+(week|month|year|\w+day)|"
        rf"{_NUM_WORDS}\s+{_TIME_UNITS}\s+ago|"
        rf"in\s+{_NUM_WORDS}\s+{_TIME_UNITS}|"
        rf"since\s+|before\s+|after\s+|until\s+|by\s+|when\s+did|"
        rf"as\s+of\s+|how\s+long\s+ago|"
        rf"recently|earlier|previously"
        rf")\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b("
        r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
        r"january|february|march|april|june|july|august|september|"
        r"october|november|december|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        r")\b",
        re.IGNORECASE,
    ),
    # 4-digit year, ISO date, or N/N/YY patterns.
    re.compile(r"\b(19|20)\d{2}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
)


class TemporalAnchor(BaseModel):
    """Validated LLM output."""

    model_config = ConfigDict(frozen=True)

    anchor: str | None = None
    reasoning: str = ""


def is_temporal_query(text: str) -> bool:
    """Cheap regex gate: does the question contain a temporal cue?"""
    return any(p.search(text) for p in _TEMPORAL_CUES)


def load_temporal_prompt() -> str:
    pkg = resources.files("engram.retrieve.prompts")
    return (pkg / TEMPORAL_PROMPT_FILENAME).read_text(encoding="utf-8")


def render_temporal_prompt(query: str, now: datetime) -> str:
    template = load_temporal_prompt()
    return template.replace("{query}", _inline(query)).replace(
        "{now}", now.isoformat()
    )


def parse_temporal_response(text: str) -> TemporalAnchor:
    payload = _strip_code_fence(text).strip()
    if not payload:
        raise AbstractionParseError("empty temporal response")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AbstractionParseError(f"invalid temporal JSON: {exc.msg}") from exc
    try:
        return TemporalAnchor.model_validate(data)
    except ValidationError as exc:
        raise AbstractionParseError(f"temporal schema mismatch: {exc}") from exc


def compute_temporal_anchor(
    query: str,
    chat: ChatProvider,
    *,
    now: datetime | None = None,
    max_retries: int = 1,
) -> datetime | None:
    """Return the anchor datetime for `query`, or None if non-temporal
    or unparseable.

    The cheap regex gate `is_temporal_query` runs first; if False, no
    chat call happens. Otherwise the LLM is asked for a JSON anchor.
    Fails open on every error path -- returning None means "no
    temporal filter."
    """
    if not is_temporal_query(query):
        return None
    if now is None:
        now = datetime.now(tz=timezone.utc)
    prompt = render_temporal_prompt(query, now)
    messages: list[Message] = [Message(role="user", content=prompt)]
    last_response = ""
    for _ in range(max_retries + 1):
        try:
            last_response = chat.chat(messages)
        except Exception:
            return None
        try:
            anchor_obj = parse_temporal_response(last_response)
            break
        except AbstractionParseError:
            messages = [
                *messages,
                Message(role="assistant", content=last_response),
                Message(
                    role="user",
                    content=(
                        "Your previous response was not valid JSON. "
                        "Respond only with the JSON object."
                    ),
                ),
            ]
    else:
        return None
    if anchor_obj.anchor is None:
        return None
    try:
        parsed = datetime.fromisoformat(anchor_obj.anchor)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # Caller asked for ISO-8601 UTC in the prompt; if the LLM
        # omitted the zone we treat the value as UTC.
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        # Normalize any tz-aware datetime to UTC.  An LLM may emit
        # '+05:30' or 'Z' or '+00:00' — semantically identical but
        # they compare unequal as datetimes (different tzinfo
        # objects), which breaks downstream cache keys and equality
        # tests that work on the anchor.
        parsed = parsed.astimezone(timezone.utc)
    return parsed


_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text


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
    "TEMPORAL_PROMPT_VERSION",
    "TemporalAnchor",
    "compute_temporal_anchor",
    "is_temporal_query",
    "parse_temporal_response",
    "render_temporal_prompt",
]
