"""No-op suite — exists to verify the harness end-to-end.

The CI smoke benchmark runs this suite. It exercises:
  - suite loading by name,
  - setup / run / teardown lifecycle,
  - manifest writing.

It does *not* exercise any Engram algorithm; that's by design.
"""

from __future__ import annotations

from engram.bench._provider import Provider
from engram.bench._suite import SuiteResult


class NoopSuite:
    name: str = "noop"
    dataset_version: str = "noop-1"
    # SHA-256 of the empty bytes (b"") is a deterministic stand-in.
    dataset_checksum: str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def __init__(self) -> None:
        self._provider: Provider | None = None

    def setup(self, provider: Provider) -> None:
        self._provider = provider

    def run(self) -> SuiteResult:
        return SuiteResult(
            name=self.name,
            aggregate_metrics={"noop": 1.0},
            confidence_intervals={"noop": (1.0, 1.0)},
            per_question=[],
            latency_ms={},
        )

    def teardown(self) -> None:
        self._provider = None


SUITE: NoopSuite = NoopSuite()
