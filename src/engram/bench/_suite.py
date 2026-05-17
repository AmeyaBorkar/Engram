"""Suite protocol and result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from engram.bench._provider import Provider


@dataclass(frozen=True)
class SuiteResult:
    """What a suite returns from `run()`.

    `aggregate_metrics` is the headline number(s); `confidence_intervals`
    holds the bootstrap CIs (n=1000 by convention) for those numbers; the
    other fields support drilling into per-question behavior and per-API
    latency.

    ``suite_metadata`` is an opt-in slot for the suite to surface
    structural facts the runner should fold into the manifest --
    prompt template versions (M-154), judge model identity, dataset
    sub-split labels, etc. The runner merges it into the manifest's
    ``engram_config`` so the SCOREBOARD column can refer to it.
    """

    name: str
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: dict[str, list[float]] = field(default_factory=dict)
    suite_metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Suite(Protocol):
    """A benchmark suite.

    Implementations live in `engram.bench.suites.<name>` (built-ins) or
    `benchmarks.suites.<name>` (project-local). The runner imports the module
    and looks for a module-level `SUITE` attribute.
    """

    name: str
    dataset_version: str
    dataset_checksum: str

    def setup(self, provider: Provider) -> None: ...
    def run(self) -> SuiteResult: ...
    def teardown(self) -> None: ...
