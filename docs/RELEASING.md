# Releasing Engram

End-to-end recipe for cutting a new release. Currently used for v0.1.0;
the same steps apply for v0.1.x patch releases and beyond.

> **PyPI name vs import name.** Engram is published on PyPI as
> `engram-memory` (the bare `engram` name is squatted as a placeholder
> by a third party; a PEP 541 reclaim is pending). The Python import
> name is `engram` -- users `pip install engram-memory` and
> `import engram`. Don't confuse the two when writing release notes.

---

## Pre-flight (every release)

```powershell
# Working tree must be clean.
git status

# Lint + format + types.
ruff check .
ruff format --check .
python -m mypy

# Full test suite (excludes slow-marked perf tests by default).
python -m pytest -q

# Slow-marked tests too. Fails here block the release.
python -m pytest -q -m slow

# Smoke benchmark via the harness.
python -m engram.bench run noop --provider fake --runs-dir benchmarks/runs/ci

# The full benchmark suite (latency + recall lift). These have their
# own slow-marked perf tests inside; the above `-m slow` already runs them.
```

All must pass before the release tag goes up.

---

## v0.1.0 release receipt

This is what landed on 2026-05-11:

  * **LongMemEval-S, 500 questions:** 71.4% (357/500)
    * Manifest: `benchmarks/runs/release/20260511T052920_486768+0000-0b6dfa53-longmemeval.json`
    * Per-type: 94.6% single-session-assistant, 84.3% single-session-user,
      72.2% temporal-reasoning, 69.2% knowledge-update, 60.2% multi-session,
      50.0% single-session-preference.
  * Stack: `BAAI/bge-large-en-v1.5` (local, GPU) embedder + Kimi K2.6 via
    OpenCode Go for both answer and judge.
  * Caveat: same-model judge (self-preference bias ~3-7pp). v0.1.1 will
    re-judge with GPT-4o.

---

## Build + dry-run on TestPyPI

```powershell
# Clean any old artefacts.
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue

# Build wheel + sdist.
python -m build

# Verify metadata. Both artefacts must PASS.
python -m twine check dist/*
```

Expected output:

```
Checking dist/engram_memory-0.1.0-py3-none-any.whl: PASSED
Checking dist/engram_memory-0.1.0.tar.gz: PASSED
```

(Optional, recommended for first release of a new name) **TestPyPI dry
run:**

```powershell
# Requires a TestPyPI account + API token (separate from PyPI).
# Token goes in ~/.pypirc or pass via --username __token__ --password <token>.
python -m twine upload --repository testpypi dist/*
```

Then verify the install works from TestPyPI in a fresh venv:

```powershell
python -m venv \tmp\engram-install-test
\tmp\engram-install-test\Scripts\Activate.ps1
pip install --index-url https://test.pypi.org/simple/ `
            --extra-index-url https://pypi.org/simple/ `
            engram-memory==0.1.0
python -c "import engram; print(engram.__version__)"
deactivate
```

If `import engram` works and prints the right version, we're good.

---

## Push to real PyPI

```powershell
python -m twine upload dist/*
```

Authenticates against `~/.pypirc` or asks interactively. Use an API
token (`__token__` / `pypi-...`), never a password.

Verify the page:
  * <https://pypi.org/project/engram-memory/>
  * Markdown rendering is sane
  * Wheel + sdist both listed under "Download files"

---

## Tag + GitHub release

```powershell
git tag -a v0.1.0 -m "v0.1.0: LongMemEval-S 71.4%"
git push origin v0.1.0

# Optional GitHub release with the manifest attached.
# Requires `gh` CLI authenticated to the repo.
gh release create v0.1.0 `
  --title "v0.1.0 - LongMemEval-S 71.4%" `
  --notes-file CHANGELOG.md `
  benchmarks/runs/release/20260511T052920_486768+0000-0b6dfa53-longmemeval.json
```

(Adjust the manifest filename to the canonical release-run path.)

---

## Post-release checklist

  * Update README install command if it changed (`pip install engram-memory`).
  * Bump `version` in `pyproject.toml` and `src/engram/__init__.py` to
    `0.1.1.dev0` (or whatever the next dev cycle uses) so subsequent
    commits don't accidentally re-publish 0.1.0.
  * Announce: README badge, social posts, etc.

---

## PEP 541 reclaim for `engram`

The bare `engram` name on PyPI is a placeholder owned by another user
("Do not use. Placeholder."). PEP 541 covers exactly this case.

File the reclaim request at
<https://github.com/pypi/support/issues/new?template=name-request.md>
citing:

  * PEP 541 (legitimate-use takeover, abandoned-name clause).
  * Evidence the existing project is unused (placeholder description,
    no real release content).
  * Engram-memory is the active, maintained project intending to use
    the name in good faith.

Typical resolution: 1-3 months. If granted, we'll publish v0.2.0 on
both `engram` and `engram-memory` (with `engram` becoming canonical),
then deprecate `engram-memory` after a one-release notice period.
