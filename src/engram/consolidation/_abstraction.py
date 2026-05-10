"""Abstraction extraction.

Given a cluster of related event contents, ask the chat provider to
articulate the general pattern they share. The result is parsed into a
strict Pydantic schema and validated; any deviation is a hard error
(retried once with a clarifying nudge before giving up).

Prompt versioning: prompts live in `engram.consolidation.prompts/` as
`<name>_v<n>.txt` files. The version is part of the prompt's
fingerprint and is recorded on every consolidated `MemoryItem`'s
metadata so downstream consumers can replay or audit.

The prompt is hardened against prompt injection by:
  1. Explicit framing of observations as data, not instructions.
  2. A negative-example clause forbidding the abstraction text from
     containing instruction-like content.
  3. Strict JSON-only output, parsed with Pydantic; any free-form
     text is treated as a parse failure.

Stage 5 also runs the existing prompt-injection corpus through this
pipeline as a regression suite (separate test file).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from engram.providers._message import Message
from engram.providers._protocols import ChatProvider

PROMPT_NAME = "abstract"
PROMPT_VERSION = "v1"
PROMPT_FILENAME = f"{PROMPT_NAME}_{PROMPT_VERSION}.txt"


# Public registry: external code that wants to pin a specific prompt
# version reads this dict. We add an entry on every version bump.
PROMPT_VERSIONS: dict[str, str] = {
    PROMPT_NAME: PROMPT_VERSION,
    "judge": "v1",
}


class AbstractionParseError(ValueError):
    """Raised when the chat response cannot be coerced into the schema."""


class AbstractionResult(BaseModel):
    """Validated output of one abstraction call.

    `abstraction` is the natural-language generalization. `confidence`
    is the LLM's self-assessed reliability. `supports` lists the
    indices of observations the LLM judged most load-bearing - the
    engine uses these to weight the provenance links.
    """

    model_config = ConfigDict(frozen=True)

    abstraction: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0)
    supports: tuple[int, ...] = Field(default=())


@dataclass(frozen=True, slots=True)
class AbstractionRequest:
    """Inputs to one abstraction call.

    `cohesion_hint` is the cluster's cohesion score (0-1) - we surface
    it in the prompt so the LLM can calibrate its `confidence` against
    how tight the cluster actually was.
    """

    observations: tuple[str, ...]
    cohesion_hint: float

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("at least one observation is required")
        if not 0.0 <= self.cohesion_hint <= 1.0:
            raise ValueError(f"cohesion_hint must be in [0, 1], got {self.cohesion_hint!r}")


def load_prompt_template() -> str:
    """Read the abstraction prompt template from the prompts package."""
    pkg = resources.files("engram.consolidation.prompts")
    return (pkg / PROMPT_FILENAME).read_text(encoding="utf-8")


def render_prompt(request: AbstractionRequest) -> str:
    """Render the abstraction prompt for one cluster.

    Observations are numbered 0..N-1 (matching the indices the LLM
    returns in `supports`). Newlines inside observations are escaped to
    `\\n` literals so each observation occupies a single line in the
    prompt - that makes the data/instructions boundary visually
    obvious to the LLM and harder to confuse.
    """
    template = load_prompt_template()
    numbered = "\n".join(
        f"{i}. {_inline(content)}" for i, content in enumerate(request.observations)
    )
    return template.replace("{cohesion}", f"{request.cohesion_hint:.3f}").replace(
        "{observations}", numbered
    )


def parse_response(text: str, n_observations: int) -> AbstractionResult:
    """Parse a chat response into `AbstractionResult`, strictly.

    Accepts the response with or without a surrounding markdown code
    fence (` ```json ... ``` `) - many providers add one even when
    asked not to. Any other surrounding prose is a parse error.

    Last line of defense: if the produced `abstraction` text matches a
    known prompt-injection pattern (e.g. "ignore previous instructions",
    "<|im_start|>", "system prompt"), the response is rejected. This
    runs even on a successful schema match - the model can echo
    injection text inside a perfectly valid JSON envelope, and we still
    do not want it in the memory hierarchy.
    """
    # Local import keeps the security check optional from a packaging
    # standpoint - downstream forks can swap the corpus without
    # rewiring `_abstraction.py`.
    from engram._security.prompt_injection import looks_like_injection

    payload = _strip_code_fence(text).strip()
    if not payload:
        raise AbstractionParseError("empty response")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AbstractionParseError(f"invalid JSON: {exc.msg}") from exc
    try:
        result = AbstractionResult.model_validate(data)
    except ValidationError as exc:
        raise AbstractionParseError(f"schema mismatch: {exc}") from exc
    for idx in result.supports:
        if not 0 <= idx < n_observations:
            raise AbstractionParseError(f"support index {idx} out of range [0, {n_observations})")
    if looks_like_injection(result.abstraction):
        raise AbstractionParseError("abstraction contains prompt-injection-like content; rejecting")
    return result


def extract_abstraction(
    request: AbstractionRequest,
    chat: ChatProvider,
    *,
    max_retries: int = 1,
) -> AbstractionResult:
    """Send the abstraction request to `chat` and parse the response.

    On parse failure, retries up to `max_retries` times with a
    clarifying nudge. Re-raises the parse error if all attempts fail.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")

    rendered = render_prompt(request)
    base: list[Message] = [Message(role="user", content=rendered)]
    last_error: AbstractionParseError | None = None
    last_response: str = ""
    for attempt in range(max_retries + 1):
        if attempt == 0:
            messages = base
        else:
            # Nudge: feed the previous bad output back and ask for
            # strict JSON only.
            messages = [
                *base,
                Message(role="assistant", content=last_response),
                Message(
                    role="user",
                    content=(
                        "Your previous response was not valid JSON matching the schema. "
                        "Respond only with the JSON object, no surrounding prose."
                    ),
                ),
            ]
        last_response = chat.chat(messages)
        try:
            return parse_response(last_response, len(request.observations))
        except AbstractionParseError as exc:
            last_error = exc
            continue
    # Loop above always sets `last_error` before reaching here (we entered
    # the loop with `max_retries + 1 >= 1` and only fall through on parse
    # failure, which always assigns).
    raise last_error  # type: ignore[misc]


_FENCE_RE = re.compile(
    r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$",
    flags=re.DOTALL,
)


def _strip_code_fence(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _inline(content: str) -> str:
    """Collapse newlines and tabs so each observation is one prompt line."""
    return content.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
