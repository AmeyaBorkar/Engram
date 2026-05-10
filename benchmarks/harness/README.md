# Benchmark harness

Reserved for the harness that drives all Engram benchmark runs. Implementation lands at Stage 1 — see `../../ROADMAP.md`.

## Design intent

- One CLI entry point: `python -m engram.bench run <suite> --config <path>`.
- Suites are pluggable; each suite implements a small protocol (`load_dataset`, `run_question`, `score`).
- Two modes: `--provider fake` (deterministic, runs in CI on every PR) and `--provider <real>` (release-only, opt-in, costs money).
- Every run emits a manifest to `../runs/<date>-<short-sha>-<suite>.json` containing:
  - environment (commit, dirty flag, Python, OS, CPU, RAM),
  - config (Engram parameters, provider config or fake hash),
  - dataset version + checksum,
  - per-question scores,
  - aggregate metrics with bootstrap CIs (n = 1000),
  - latency histograms.
- Manifests are committed; results without a manifest don't count toward `../SCOREBOARD.md`.

## Non-goals

- Replacing third-party leaderboards. We track *our* number under *our* discipline; we don't publish a competing leaderboard.
- Running real-provider benchmarks in CI on every PR. Cost-prohibitive and noisy. The fake-provider smoke run guards against framework regressions; release runs guard against algorithmic regressions.
