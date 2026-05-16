"""Shared LLM prompt-render and parse helpers.

Three responsibilities, each centralized so a security fix lands in one
place instead of nine modules:

  * `inline(content)` — collapse line-breaking characters into `\\n` so
    one observation is one prompt line.  Closes injection vectors that
    rely on the model interpreting a raw newline / CR / U+2028 /
    U+2029 / tab as a structural break.  Backslash is escaped first
    so the escape sequences survive.

  * `strip_code_fence(text)` — pull JSON out of a `` ```json … ``` ``
    fence; passes through unfenced text unchanged.  Used after every
    LLM-output JSON parse step.

  * `render_prompt(template, **fields)` — single-pass substitution of
    `{name}` placeholders.  Unlike `t.replace("{a}",a).replace("{b}",b)`
    (which lets attacker-controlled text in `a` masquerade as a literal
    `{b}` and steer the second `replace` into the wrong slot), this
    walks the template once and substitutes each placeholder at most
    once.  Placeholders with no field provided are left literal.

This module is import-cheap (stdlib `re` only) so every prompt-render
path can import it without circulars.
"""

from __future__ import annotations

import re
from typing import Mapping


def inline(content: str) -> str:
    """Collapse line-breaking characters so user content is one prompt line.

    Order matters: backslash MUST be escaped first or its escape
    sequences would themselves be escaped.  CRLF is collapsed before
    LF/CR so a Windows `\\r\\n` becomes one `\\n`, not two.
    """
    return (
        content.replace("\\", "\\\\")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
        .replace(" ", "\\n")
        .replace(" ", "\\n")
        .replace("\t", "\\t")
    )


_FENCE_RE = re.compile(
    r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$",
    flags=re.DOTALL,
)


def strip_code_fence(text: str) -> str:
    """Pull JSON out of a fenced block; pass through unfenced text.

    Matches ``` ```json ... ``` ```, ``` ```JSON ... ``` ```, or a bare
    ``` ``` ... ``` ``` fence.  Matches at the start AND end of the
    stripped text, so a fence with prose preamble falls through to the
    raw text and the caller's JSON parser will surface the real error.
    """
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def render_prompt(template: str, fields: Mapping[str, str] | None = None, /, **kwargs: str) -> str:
    """Single-pass substitution of `{name}` placeholders.

    Accepts either a mapping or kwargs.  Both forms are merged; kwargs
    win on collision.  Substitution walks the template ONCE so a value
    that happens to contain `{other_field}` cannot redirect into the
    other field's slot.

    Placeholders not present in `fields`/`kwargs` are left literal
    (no `KeyError`), preserving the existing prompt-render semantics
    where partial-render is common.
    """
    merged: dict[str, str] = dict(fields or {})
    merged.update(kwargs)
    if not merged:
        return template
    keys_pattern = "|".join(re.escape(k) for k in merged)
    placeholder_re = re.compile(r"\{(" + keys_pattern + r")\}")
    return placeholder_re.sub(lambda m: merged[m.group(1)], template)


__all__ = ["inline", "render_prompt", "strip_code_fence"]
