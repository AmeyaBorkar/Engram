"""Verification pass for `EngramAgent`.

After the initial chat call, ask the chat provider whether the
candidate answer is supported by the retrieved context. If not, the
EngramAgent re-retrieves (with a refined query if available) and
retries up to `verify_max_retries` times. Catches the cases where
the LLM hallucinated despite having grounded context.

Cost: one extra chat call per retry. The bias on parse failure is
`supported=True` -- a broken verifier shouldn't loop forever.
"""

from __future__ import annotations

import json
import re
from importlib import resources

from pydantic import BaseModel, ConfigDict, ValidationError

from engram.consolidation._abstraction import AbstractionParseError
from engram.providers._message import Message
from engram.providers._protocols import ChatProvider

VERIFY_PROMPT_NAME = "verify"
VERIFY_PROMPT_VERSION = "v1"
VERIFY_PROMPT_FILENAME = f"{VERIFY_PROMPT_NAME}_{VERIFY_PROMPT_VERSION}.txt"


class VerifyVerdict(BaseModel):
    """Verifier output."""

    model_config = ConfigDict(frozen=True)

    supported: bool
    reason: str = ""


def load_verify_prompt() -> str:
    pkg = resources.files("engram.integrations.prompts")
    return (pkg / VERIFY_PROMPT_FILENAME).read_text(encoding="utf-8")


def render_verify_prompt(*, question: str, context: str, answer: str) -> str:
    template = load_verify_prompt()
    return (
        template.replace("{question}", _inline(question))
        .replace("{context}", context if context else "(none)")
        .replace("{answer}", _inline(answer))
    )


def parse_verify_response(text: str) -> VerifyVerdict:
    payload = _strip_code_fence(text).strip()
    if not payload:
        raise AbstractionParseError("empty verify response")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AbstractionParseError(f"invalid verify JSON: {exc.msg}") from exc
    try:
        return VerifyVerdict.model_validate(data)
    except ValidationError as exc:
        raise AbstractionParseError(f"verify schema mismatch: {exc}") from exc


def verify_answer(
    *,
    question: str,
    context: str,
    answer: str,
    chat: ChatProvider,
    max_retries: int = 1,
) -> VerifyVerdict:
    """Ask the chat provider if `answer` is supported by `context`.

    On any parse failure across retries, returns
    `(supported=True, reason="")` so the caller exits the loop. The
    bias is "stop retrying" -- a malfunctioning verifier shouldn't
    drive infinite re-asks.
    """
    prompt = render_verify_prompt(question=question, context=context, answer=answer)
    messages: list[Message] = [Message(role="user", content=prompt)]
    last_response = ""
    for _ in range(max_retries + 1):
        last_response = chat.chat(messages)
        try:
            return parse_verify_response(last_response)
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
    return VerifyVerdict(supported=True, reason="")


_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def _inline(content: str) -> str:
    return content.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


__all__ = [
    "VERIFY_PROMPT_VERSION",
    "VerifyVerdict",
    "parse_verify_response",
    "render_verify_prompt",
    "verify_answer",
]
