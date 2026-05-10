"""Engram benchmark harness.

Stage 1 lays the framework: a CLI entry point, a `Suite` protocol, a
provider stub, and a manifest writer. Suites and baselines are layered on
in later stages — see `ROADMAP.md` and `benchmarks/SOTA.md`.
"""

from engram.bench._cli import main
from engram.bench._manifest import Manifest, gather_environment, manifest_from_run
from engram.bench._provider import FakeProvider, Provider
from engram.bench._retriever import EngramRetriever, Hit, Retriever
from engram.bench._runner import load_suite, run
from engram.bench._suite import Suite, SuiteResult

__all__ = [
    "EngramRetriever",
    "FakeProvider",
    "Hit",
    "Manifest",
    "Provider",
    "Retriever",
    "Suite",
    "SuiteResult",
    "gather_environment",
    "load_suite",
    "main",
    "manifest_from_run",
    "run",
]
