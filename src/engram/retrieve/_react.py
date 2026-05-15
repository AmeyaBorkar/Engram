"""Iterative ReAct-style retrieval.

The standard retrieve is one-shot: embed the query, find top-k,
return. For complex multi-hop or knowledge-update questions, the
first hop's results may not contain the answer -- but they may
contain enough hints to refine the query into something that does.

ReAct retrieval [Yao et al. 2022, arXiv:2210.03629] interleaves
retrieval and reasoning: retrieve -> judge whether enough is in
hand -> refine query -> retrieve again, up to `max_steps`. The
judge prompt asks the LLM a closed-form decision: sufficient
(yes/no) + refined_query (short search phrase).

Cost: one chat call per step. Worth it on the multi-hop /
multi-session splits where the cost is the LLM's, not the
retrieval system's.
"""

from __future__ import annotations

import json
import re
from importlib import resources

from pydantic import BaseModel, ConfigDict, ValidationError

from engram.consolidation._abstraction import AbstractionParseError
from engram.providers._message import Message
from engram.providers._protocols import ChatProvider
from engram.schemas import RetrievalResult

REACT_PROMPT_NAME = "react"
REACT_PROMPT_VERSION = "v1"
REACT_PROMPT_FILENAME = f"{REACT_PROMPT_NAME}_{REACT_PROMPT_VERSION}.txt"


class ReactVerdict(BaseModel):
    """LLM's per-step judgment."""

    model_config = ConfigDict(frozen=True)

    sufficient: bool
    refined_query: str = ""


def load_react_prompt() -> str:
    pkg = resources.files("engram.retrieve.prompts")
    return (pkg / REACT_PROMPT_FILENAME).read_text(encoding="utf-8")


def render_react_prompt(question: str, memories: list[RetrievalResult]) -> str:
    template = load_react_prompt()
    if not memories:
        memory_block = "(none yet)"
    else:
        lines = [f"- [{r.level.value}] {r.content}" for r in memories]
        memory_block = "\n".join(lines)
    return template.replace("{question}", _inline(question)).replace(
        "{memories}", memory_block
    )


def parse_react_response(text: str) -> ReactVerdict:
    payload = _strip_code_fence(text).strip()
    if not payload:
        raise AbstractionParseError("empty react response")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AbstractionParseError(f"invalid react JSON: {exc.msg}") from exc
    try:
        return ReactVerdict.model_validate(data)
    except ValidationError as exc:
        raise AbstractionParseError(f"react schema mismatch: {exc}") from exc


def react_judge(
    question: str,
    memories: list[RetrievalResult],
    chat: ChatProvider,
    *,
    max_retries: int = 1,
) -> ReactVerdict:
    """Run the judge for one step.

    On any parse failure across retries, returns `(sufficient=True,
    refined_query="")` so the caller exits the loop gracefully. The
    fallback bias is "stop iterating" -- a malfunctioning judge
    shouldn't drive infinite retrieval.
    """
    prompt = render_react_prompt(question, memories)
    messages: list[Message] = [Message(role="user", content=prompt)]
    last_response = ""
    for _ in range(max_retries + 1):
        try:
            last_response = chat.chat(messages)
        except Exception:
            # Sibling helpers (decompose, hyde, multi_query, temporal) all
            # treat a chat-provider failure as best-effort: log/swallow and
            # return a sensible fallback.  react_judge previously let the
            # exception escape, killing the iterative retrieve loop — and
            # the surrounding Memory.retrieve_iterative call — on the
            # first transient network blip.  Match the sibling pattern
            # and gracefully exit the loop instead.
            return ReactVerdict(sufficient=True, refined_query="")
        try:
            return parse_react_response(last_response)
        except AbstractionParseError:
            messages = [
                *messages,
                Message(role="assistant", content=last_response),
                Message(
                    role="user",
                    content=(
                        "Your previous response was not valid JSON matching "
                        "the schema. Respond only with the JSON object."
                    ),
                ),
            ]
    return ReactVerdict(sufficient=True, refined_query="")


_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def _inline(content: str) -> str:
    return content.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


__all__ = [
    "REACT_PROMPT_VERSION",
    "ReactVerdict",
    "parse_react_response",
    "react_judge",
    "render_react_prompt",
]
