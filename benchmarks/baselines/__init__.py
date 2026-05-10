"""Baseline retriever adapters used in `benchmarks/SOTA.md`.

Each adapter implements `engram.bench.Retriever`. They are NOT installed
as part of the engram package; live next to the benchmarks they support
and are gated behind the `[bench]` extra so they don't pull heavy deps
into the core install.
"""
