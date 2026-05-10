# LongMemEval runbook

Step-by-step instructions for running the LongMemEval-S benchmark
against Engram with real LLM providers (OpenAI, Anthropic, Moonshot/Kimi).

The dataset is research-licensed and **not vendored** in this repo. The
download script in `scripts/fetch_longmemeval.py` pulls it from
HuggingFace into `benchmarks/datasets/`, which is gitignored.

---

## 1. Install the right extras

```powershell
pip install -e ".[dev,openai,anthropic]"
```

`openai` is used as the embedder for every real-provider run (Anthropic
and Moonshot don't ship public embedding models). `anthropic` is
optional unless you plan to judge with Claude. Moonshot/Kimi reuses the
OpenAI extra (Moonshot's API is OpenAI-compatible).

---

## 2. Set up API keys

Engram reads keys from environment variables. Pick the ones you need:

| Variable | Provider | Where to get one |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI (embeddings + chat) | <https://platform.openai.com/api-keys> |
| `ANTHROPIC_API_KEY` | Anthropic (chat only) | <https://console.anthropic.com/settings/keys> |
| `MOONSHOT_API_KEY` | Moonshot/Kimi K2 (chat only) | <https://platform.moonshot.ai/> |

PowerShell (current session only):

```powershell
$env:OPENAI_API_KEY    = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:MOONSHOT_API_KEY  = "ms-..."
```

To persist across sessions, set the same variables in
`Settings → System → Environment Variables` (User scope) or via
`[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-...", "User")`.

> Engram never logs API keys. Provider adapters apply the
> `Redactor` pass before any structured log emission; CLI flags are
> never echoed back into manifests.

---

## 3. Download the dataset

```powershell
python scripts/fetch_longmemeval.py            # default: longmemeval_s (500 questions, ~265 MiB)
python scripts/fetch_longmemeval.py --split m  # ~5,000 sessions per question (much larger)
python scripts/fetch_longmemeval.py --split oracle
```

The file lands at `benchmarks/datasets/longmemeval/longmemeval_s_cleaned.json`.
Re-running the script is a no-op when the file already exists; pass
`--force` to overwrite.

The fetch logs the SHA-256 of the file -- the bench harness records
the same hash in every manifest, so you can verify which version of
the dataset produced which result.

---

## 4. Smoke run (10 questions, ~$0.02)

Always do a smoke run first. It catches:

  * Missing API keys (clear `RuntimeError` instead of mid-run failure).
  * Quota / rate-limit issues with your account.
  * Per-question prompt rendering bugs.

```powershell
$env:LONGMEMEVAL_MAX_QUESTIONS = "10"
python -m engram.bench run longmemeval `
  --embedder openai `
  --chat openai `
  --chat-model gpt-4o-mini `
  --runs-dir benchmarks/runs/local
```

You'll see one line per 10 questions logging the running accuracy.
The manifest is written under `benchmarks/runs/local/`.

Inspect:

```powershell
python -c "import json; from pathlib import Path; m = sorted(Path('benchmarks/runs/local').glob('*longmemeval*.json'))[-1]; d = json.loads(m.read_text()); print(d['aggregate_metrics'])"
```

---

## 5. Full run (500 questions, ~$1)

Clear the smoke cap and run the full split:

```powershell
$env:LONGMEMEVAL_MAX_QUESTIONS = ""
python -m engram.bench run longmemeval `
  --embedder openai `
  --chat openai `
  --chat-model gpt-4o-mini `
  --runs-dir benchmarks/runs/local
```

Wallclock varies with rate limits; expect roughly 30–60 minutes for
500 questions on `gpt-4o-mini`. The manifest captures per-type
accuracy plus the per-question rows so you can drill into failures.

### Variants

**Stronger judge (more expensive but truer to LongMemEval's reported numbers):**

```powershell
python -m engram.bench run longmemeval `
  --embedder openai `
  --chat openai `
  --chat-model gpt-4o `
  --runs-dir benchmarks/runs/local
```

**Kimi K2 as the answer model (judging with OpenAI):**

The current suite uses one chat provider for both answer-generation and
the judge. Kimi-as-answerer + OpenAI-as-judge is the next step, but for
the first run you can swap them as a unit:

```powershell
python -m engram.bench run longmemeval `
  --embedder openai `
  --chat moonshot `
  --chat-model kimi-k2.6 `
  --runs-dir benchmarks/runs/local
```

**Anthropic Claude as the answer model:**

```powershell
python -m engram.bench run longmemeval `
  --embedder openai `
  --chat anthropic `
  --chat-model claude-haiku-4-5-20251001 `
  --runs-dir benchmarks/runs/local
```

---

## 6. Update the SCOREBOARD

After the full run, copy the manifest path and the headline numbers
into `benchmarks/SCOREBOARD.md`. The relevant rows:

  * **LongMemEval** table: `Engram (target, v0.1)` -> the run's
    `aggregate_metrics["accuracy"]`.
  * Add a "Best public" row for the cited LongMemEval paper.

The manifest path goes in the right-hand "Manifest" column so anyone
verifying the result can re-run from the recorded git commit + dataset
checksum + provider hash.

---

## Cost estimates (LongMemEval-S, ~500 questions)

| Provider combination | Approx total |
|---|---|
| OpenAI `gpt-4o-mini` (answer + judge) + `text-embedding-3-small` | ~$1 |
| OpenAI `gpt-4o` (answer + judge) + `text-embedding-3-small` | ~$8 |
| Moonshot `kimi-k2.6` (answer + judge) + OpenAI `text-embedding-3-small` | ~$2 |
| Anthropic `claude-haiku` (answer + judge) + OpenAI `text-embedding-3-small` | ~$3 |

Numbers are rough and vary with rate-limit-induced retries and cache
hits. The manifest's per-call latency histograms tell you the real
distribution after the run.
