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
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),                  # Anthropic API key
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),                       # OpenAI / generic sk- key
    re.compile(r"AKIA[0-9A-Z]{16}"),                             # AWS access key id
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),                # Bearer tokens
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
        """Build a redactor from string or pre-compiled regex patterns."""
        compiled = tuple(p if isinstance(p, re.Pattern) else re.compile(p) for p in patterns)
        return cls(patterns=compiled, replacement=replacement)

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
