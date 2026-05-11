"""Preference-statement detection (Stage 9 / E.6).

A heuristic that flags text as preference-bearing. Used by callers
that want to auto-route preference statements into the dedicated
`Level.PREFERENCE` layer. Memory.record_preference uses it as the
public escape hatch; users can also call `is_preference` directly
to gate their own observe path.

Why heuristic, not LLM: detection has to be cheap. The cost of a
false positive is small (preference layer gets a non-preference item;
no big deal). The cost of a false negative is a single layer-miss; the
item still lands in the default hierarchy. So a simple regex
ensemble that captures common phrasings is the right shape.

Patterns covered:
  * First-person preference verbs:
    "I love/like/prefer/enjoy/hate/dislike/adore/can't stand X"
  * Habitual / aspectual: "I always/never/usually/typically X"
  * Possessive favorites: "my favorite X is Y"
  * Comparative preference: "I'd rather X than Y"
  * Fandom phrasings: "I'm a (huge|big) fan of X"
"""

from __future__ import annotations

import re

# Common adverb modifiers that can appear between "I" and the verb
# ("I really like", "I just love", "I genuinely prefer", ...).
_ADV = r"(?:really|truly|just|generally|usually|absolutely|honestly|genuinely|always|never)\s+"

_PREFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\bi\s+(?:{_ADV})?(love|like|prefer|enjoy|adore|hate|dislike|loathe)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bi\s+can'?t\s+stand\b", re.IGNORECASE),
    re.compile(
        r"\bi\s+(always|never|usually|typically|generally|tend\s+to)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bmy\s+favo(u)?rite\b", re.IGNORECASE),
    re.compile(r"\bi'?d?\s+rather\b", re.IGNORECASE),
    re.compile(r"\bi'?m\s+a\s+(huge|big|massive)?\s*fan\s+of\b", re.IGNORECASE),
    re.compile(r"\bi\s+would\s+(prefer|rather|love|hate)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+a\s+fan\s+of\b", re.IGNORECASE),
)


def is_preference(text: str) -> bool:
    """Return True if `text` looks like a preference statement.

    Pure regex; case-insensitive; ~10us per call at 200-character
    inputs. False positives are cheap (item ends up in preference
    layer unnecessarily); false negatives leave the item in the
    default hierarchy. Both are recoverable.
    """
    return any(p.search(text) for p in _PREFERENCE_PATTERNS)


__all__ = ["is_preference"]
