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
import logging
import re
from datetime import datetime, timezone
from importlib import resources

from pydantic import BaseModel, ConfigDict, ValidationError

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

_LOG = logging.getLogger("engram.retrieve")

TEMPORAL_PROMPT_NAME = "temporal_anchor"
TEMPORAL_PROMPT_VERSION = "v1"
TEMPORAL_PROMPT_FILENAME = f"{TEMPORAL_PROMPT_NAME}_{TEMPORAL_PROMPT_VERSION}.txt"


# Heuristic gate so we don't burn a chat call on every retrieve. If
# the question contains no temporal cue, skip the codegen prompt.
_NUM_WORDS = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|few|several|many|couple|dozen)"
)
_TIME_UNITS = r"(?:day|days|week|weeks|month|months|year|years|hour|hours)"

# Audit M-39: the temporal prepositions "since/before/after/until/by"
# alone are too permissive — "by hand", "after lunch", "before noon"
# all match indiscriminately and fire the temporal LLM call on
# clearly-non-temporal questions.  Require the preposition to be
# followed (within a small window) by an obvious date-shaped token:
# a digit, a 4-digit year, a month or weekday name, or one of the
# relative-time nouns ("yesterday", "today", ...).
_DATE_TOKEN = (
    r"(?:"
    r"\d+|"  # digits (one or more)
    r"yesterday|today|tomorrow|tonight|now|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|"
    r"october|november|december|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"this|last|next"
    r")"
)
# The trailing `\w*` covers natural-language suffixes that aren't
# themselves date-shaped (e.g. "5pm", "15th", "2024-03-15") so the
# outer `\b` after the alternation group still finds a word boundary.
_TEMPORAL_PREP = r"(?:since|before|after|until|by)\s+(?:the\s+)?" + _DATE_TOKEN + r"\w*"

_TEMPORAL_CUES: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\b("
        rf"yesterday|today|tomorrow|tonight|last\s+(week|month|year|night|"
        rf"\w+day)|next\s+(week|month|year|\w+day)|"
        rf"{_NUM_WORDS}\s+{_TIME_UNITS}\s+ago|"
        rf"in\s+{_NUM_WORDS}\s+{_TIME_UNITS}|"
        rf"{_TEMPORAL_PREP}|"
        rf"when\s+did|"
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
    return render_prompt(template, query=_inline(query), now=now.isoformat())


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

    Retry semantics (audit M-41): the retry loop covers both the chat
    call and the JSON parse.  A transient chat-provider exception
    (rate limit, timeout, connection reset) does NOT short-circuit the
    retry — it just counts as a failed attempt and we try again until
    `max_retries + 1` attempts have been exhausted.  Only after the
    loop runs out do we return None.

    Datetime hygiene (audit M-40): the prompt asks the LLM for
    ISO-8601 UTC.  When the model emits a NAIVE datetime (no zone
    info) we log a WARNING and treat it as UTC anyway — the alternative
    (raising) would surface as a hard retrieve failure on a soft
    LLM-shape mismatch.  Operators see the warning and can tighten
    the prompt or update the prompt version.
    """
    if not is_temporal_query(query):
        return None
    if now is None:
        now = datetime.now(tz=timezone.utc)
    prompt = render_temporal_prompt(query, now)
    messages: list[Message] = [Message(role="user", content=prompt)]
    last_response = ""
    anchor_obj: TemporalAnchor | None = None
    for attempt in range(max_retries + 1):
        try:
            last_response = chat.chat(messages)
        except Exception as exc:
            # Audit M-41: transient errors (rate limit, timeout,
            # connection reset) used to short-circuit the retry by
            # returning None immediately.  Now they count as a failed
            # attempt: we log + continue so the retry budget actually
            # gets spent.  Only the FINAL attempt's failure returns
            # None (handled after the loop via the `else` branch).
            _LOG.warning(
                "temporal anchor: chat raised %s on attempt %d/%d: %s; retrying",
                type(exc).__name__,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            continue
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
    if anchor_obj is None or anchor_obj.anchor is None:
        return None
    try:
        parsed = datetime.fromisoformat(anchor_obj.anchor)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # Audit M-40: naive datetimes are documented as wrong (prompt
        # asks for ISO-8601 UTC).  We coerce to UTC so the surface
        # keeps working but emit a warning so the operator can fix
        # the prompt / provider.
        _LOG.warning(
            "temporal anchor: LLM returned naive datetime %r; "
            "treating as UTC (prompt requests ISO-8601 with offset)",
            anchor_obj.anchor,
        )
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        # Normalize any tz-aware datetime to UTC.  An LLM may emit
        # '+05:30' or 'Z' or '+00:00' — semantically identical but
        # they compare unequal as datetimes (different tzinfo
        # objects), which breaks downstream cache keys and equality
        # tests that work on the anchor.
        parsed = parsed.astimezone(timezone.utc)
    return parsed


__all__ = [
    "TemporalAnchor",
    "compute_temporal_anchor",
    "is_temporal_query",
    "parse_temporal_response",
    "render_temporal_prompt",
]
