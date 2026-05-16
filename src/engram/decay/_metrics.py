"""Observable metrics for the decay engine.

`DecayMetrics` is a frozen snapshot a caller can scrape at any moment to
build a counter view of the system. The engine itself caches the most
recent `TickResult` so dashboards can read tick latency without
re-running the sweep.

Scope: Stage 4's DoD requires that "Reinforcement and corroboration
counters are exported as metrics." We export all three signal counters
plus the cold/hot breakdown and the most-recent tick. Stage 9 wires
this into OpenTelemetry counters/histograms; until then `metrics()` is
the canonical introspection surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from engram.schemas import ItemKind

if TYPE_CHECKING:
    from engram.decay._engine import TickResult


@dataclass(frozen=True, slots=True)
class KindCounters:
    """Per-kind aggregate of decay state.

    `hot_items` and `cold_items` are gauges (current values). The
    `*_total` fields sum the per-row counters across hot rows only -
    reading the cold pool would conflate "this item was reinforced once
    before being pruned" with "this hot item is still being reinforced",
    which is rarely what the consumer wants.
    """

    kind: ItemKind
    hot_items: int
    cold_items: int
    reinforcement_total: int
    corroboration_total: int
    contradiction_total: int


@dataclass(frozen=True, slots=True)
class DecayMetrics:
    """Snapshot of the engine's observable counters.

    Aggregates across both kinds (events + memory items). Per-kind
    breakdowns are in `per_kind`.
    """

    hot_items: int
    cold_items: int
    reinforcement_total: int
    corroboration_total: int
    contradiction_total: int
    last_tick: TickResult | None
    per_kind: dict[ItemKind, KindCounters]
