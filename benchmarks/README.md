# Benchmarks

Engram's success criterion is **beating SOTA on long-horizon memory benchmarks**. Everything in this directory exists to make that claim verifiable.

- **`SOTA.md`** — running plan: which suites, which baselines, what targets, why we believe we can win.
- **`SCOREBOARD.md`** — running comparison of best public numbers vs. Engram, refreshed each release.
- **`harness/`** — the CLI / framework that drives every run. Implementation lands at Stage 1 of `../ROADMAP.md`.
- **`baselines/`** — adapters to the systems we compete with (Chroma, Letta, Zep, Cognee, HippoRAG, mem0, A-MEM, full-context).
- **`suites/`** — per-suite loaders, scorers, and configs (LongMemEval, LoCoMo, custom procedural).
- **`runs/`** — committed manifests of every release benchmark run. A claim without a manifest doesn't count.

Reproducibility discipline, environment capture, and the rule for what constitutes a "verified" result are documented in `SOTA.md`.

Until Stage 6 ships, this directory is plan and harness only — no Engram numbers yet.
