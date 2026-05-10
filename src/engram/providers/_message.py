"""Chat message schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    """One turn in a chat conversation."""

    role: Role
    content: str
