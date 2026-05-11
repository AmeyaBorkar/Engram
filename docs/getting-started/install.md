# Install

Engram requires Python 3.10 or newer.

## Standard install

```bash
pip install engram-memory
```

The Python import name stays `engram` regardless of the distribution name:

```python
import engram  # works
```

## Optional extras

| Extra | What it brings |
|---|---|
| `[openai]` | `openai` SDK for `OpenAIEmbedder` / `OpenAIChat`. |
| `[anthropic]` | `anthropic` SDK for `AnthropicChat`. |
| `[postgres]` | `psycopg[binary]>=3.1` for the Postgres backend (v0.4.0). |
| `[sqlite-vec]` | `sqlite-vec` for native vector indexing in SQLite. |
| `[consolidation]` | `hdbscan` for density-based clustering (Stage 5). |
| `[otel]` | `opentelemetry-api` + `opentelemetry-sdk` for span/metric emission. |
| `[langgraph]` | `langgraph` for the `engram.integrations.langgraph` adapter. |
| `[llamaindex]` | `llama-index-core` for the `engram.integrations.llamaindex` adapter. |
| `[bench]` | `chromadb` + `python-dotenv` for benchmark baselines + .env loading. |
| `[dev]` | The full dev toolchain (pytest, ruff, mypy, hypothesis, all of the above). |

Install multiple extras at once:

```bash
pip install "engram-memory[openai,otel,langgraph]"
```

## Verify the install

```python
import engram
print(engram.__version__)
```
