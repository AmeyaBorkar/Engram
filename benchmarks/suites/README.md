# Benchmark suites

One subdirectory per suite. Each subdirectory contains:

- A loader for the suite's dataset (cached in `~/.cache/engram/datasets/<suite>/`, never committed).
- A scorer that produces per-question and aggregate metrics.
- A `config.toml` with default Engram parameters for that suite.
- A `README.md` with the suite's definition, source, version pin, and how its score maps to the headline number reported in `../SCOREBOARD.md`.

Planned suites (see `../SOTA.md`):

- `longmemeval/` — Wu et al., 2024.
- `locomo/` — Maharana et al., 2024.
- `procedural/` — built in-house at Stage 7.

Each suite must run end-to-end against the fake provider in under 60 seconds. If it can't, the harness needs a `--smoke` mode that subsamples; the full run is then opt-in.
