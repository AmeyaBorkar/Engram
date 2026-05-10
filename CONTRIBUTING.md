# Contributing to Engram

Engram is early. Issues, experiments, and PRs are all welcome.

## Development setup

```bash
git clone https://github.com/<your-fork>/engram
cd engram
pip install -e ".[dev]"
pytest
```

Use whichever environment manager you prefer (system Python, `uv`, `conda`, `pixi`, …). Engram targets Python 3.10+; the `dev` extra installs `pytest`, `pytest-cov`, `hypothesis`, `ruff`, and `mypy`.

Provider-specific backends are optional extras (`pip install -e ".[openai]"`, `[anthropic]`, `[postgres]`, `[duckdb]`). Core Engram has no required runtime dependency, so the test suite runs without API keys — provider tests use a deterministic fake.

## Conventions

- **Style.** `ruff format` and `ruff check`. Line length 100. Security rules (`S`) are enforced.
- **Types.** Public APIs are fully typed; `mypy --strict` is clean on `src/engram`. Internal helpers — type where it helps a reader.
- **Tests.** `pytest` with `pytest-cov` and `hypothesis`. Pure-logic tests must not require network or API keys; use the in-repo fakes when a provider is needed.
- **Commits.** Imperative mood, first line ≤ 72 chars. Reference an issue when relevant.
- **Provenance.** Memory items always retain links to their supporting events. Don't drop provenance to save space.
- **SQL.** Parameterized queries only. String concatenation into SQL is a CI-blocking lint failure.

## What "production-grade" means here

Every change holds the same bar — see `ROADMAP.md` for the full standards. In short:

- A perf budget for new public APIs (microbenchmark + assertion).
- Property-based tests for invariants (weights bounded, decay monotonic, provenance never dangles).
- No plaintext PII in logs; the redactor pass is on by default in shipped adapters.
- Determinism: components take an injectable clock and RNG. Replays are exact.

## Where to start

The most useful contributions right now (per the README):

- **Benchmark runs** — reproducing baselines and finding failure modes.
- **Algorithmic experiments** — alternative consolidation strategies, decay functions, retrieval policies.
- **Integrations** — bindings for popular agent / RAG frameworks.
- **Edge cases** — adversarial conversations or agent traces that break the current implementation.

The roadmap in `README.md` is the source of truth for what's coming.

## Reporting issues

Include: Engram version, Python version, OS, and a minimal reproducer. For algorithmic regressions, attach the input trace and the expected vs. actual retrieval output.
