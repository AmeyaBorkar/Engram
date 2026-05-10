"""Provider stub for the harness.

Stage 2 expands this into the full embedding/chat provider abstraction with
batching, retries, caching, and redaction. For Stage 1 the harness only
needs to know a provider's name and a stable hash for manifest reproducibility.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    """Minimal provider interface for the Stage 1 harness."""

    name: str

    def manifest_hash(self) -> str:
        """Stable identifier of this provider's configured behavior.

        For deterministic providers (the fake) this is a fixed string;
        for real providers it should incorporate model id, version, and any
        relevant config so that two manifests can be compared by hash.
        """


class FakeProvider:
    """Deterministic harness stub.

    Replaced by `engram.providers.FakeProvider` once Stage 2 lands; kept here
    so the harness can run end-to-end without depending on Stage 2.
    """

    name: str = "fake"

    def manifest_hash(self) -> str:
        return "fake/stage1/v1"
