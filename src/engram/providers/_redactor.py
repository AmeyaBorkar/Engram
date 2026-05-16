"""PII / secret redaction for provider request and response logging.

Engram does not log raw provider payloads; everything that flows into the
structured logger goes through a `Redactor` first. The default pattern set
covers provider API keys (OpenAI, Anthropic, AWS), Bearer tokens, emails,
phone numbers, SSNs, and credit-card-shaped digit groups.

The redactor is regex-based and intentionally errs on the side of over-
redaction: false positives that scrub legitimate text are far cheaper than
false negatives that leak a key.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# fmt: off
_DEFAULT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Order matters: more-specific provider keys win over the generic `sk-`
    # bucket, so we list them first.
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),                   # Anthropic API key
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"),                  # OpenAI project key
    re.compile(r"sk-svcacct-[A-Za-z0-9_\-]{20,}"),               # OpenAI service-account key
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),                       # OpenAI / generic sk- key
    re.compile(r"AKIA[0-9A-Z]{16}"),                             # AWS access key id
    re.compile(r"\b[A-Za-z0-9/+]{40}\b"),                        # AWS secret access key (40 b64 chars)
    re.compile(r"hf_[A-Za-z0-9]{20,}"),                          # HuggingFace token
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),                   # GitHub PAT / OAuth / refresh
    re.compile(r"xox[abprso]-[A-Za-z0-9\-]{10,}"),               # Slack tokens
    re.compile(r"co_[A-Za-z0-9]{20,}"),                          # Cohere
    # JWT shape: three dot-separated base64url chunks.  Greedy enough to
    # catch the body, narrow enough not to swallow plain prose.
    re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
    # Bearer / Authorization tokens.  Char class includes b64-url padding
    # (`=`, `+`, `/`) so the regex doesn't truncate at the first padding
    # char and leak the tail of the token.
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-+/=]+"),
    re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+"),                  # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                        # US SSN
    re.compile(r"\b(?:\d[ \-]?){13,19}\b"),                      # credit-card-shaped
    re.compile(r"\b\d{3}[\s.\-]\d{3}[\s.\-]\d{4}\b"),            # US phone (formatted)
)
# fmt: on


@dataclass
class Redactor:
    """Apply a list of regexes to text, replacing matches.

    Order matters: more specific patterns (e.g. Anthropic before generic
    `sk-`) come first so they win.
    """

    patterns: tuple[re.Pattern[str], ...] = field(default_factory=lambda: _DEFAULT_PATTERNS)
    replacement: str = "[REDACTED]"

    @classmethod
    def default(cls) -> Redactor:
        """The shipped defaults: provider keys, Bearer, email, phone, SSN, card."""
        return cls()

    @classmethod
    def from_patterns(
        cls, patterns: Iterable[str | re.Pattern[str]], replacement: str = "[REDACTED]"
    ) -> Redactor:
        """Build a redactor from string or pre-compiled regex patterns.

        Compiles each string pattern; a malformed regex raises
        ``ValueError`` naming the index and the underlying re.error so
        a JSON / YAML config with a typo is diagnosable without
        bisecting the pattern list by hand.
        """
        compiled_list: list[re.Pattern[str]] = []
        for i, p in enumerate(patterns):
            if isinstance(p, re.Pattern):
                compiled_list.append(p)
                continue
            try:
                compiled_list.append(re.compile(p))
            except re.error as exc:
                raise ValueError(
                    f"redactor pattern at index {i} ({p!r}) is invalid: {exc}"
                ) from exc
        return cls(patterns=tuple(compiled_list), replacement=replacement)

    def redact(self, text: str) -> str:
        """Return `text` with every match of any pattern replaced."""
        result = text
        for pattern in self.patterns:
            result = pattern.sub(self.replacement, result)
        return result

    def redact_obj(self, obj: Any) -> Any:
        """Walk a JSON-like object, redacting every string leaf.

        Containers (`dict`, `list`, `tuple`) are reconstructed; non-strings
        are passed through unchanged. Used for scrubbing structured log
        payloads before emission.
        """
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, dict):
            return {k: self.redact_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.redact_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.redact_obj(v) for v in obj)
        return obj
