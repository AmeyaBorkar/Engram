# Benchmarks

Reserved for benchmark code, configs, and results.

Planned suites (per the roadmap):

- **LongMemEval** — long-horizon conversational memory.
- **LoCoMo** — multi-session dialogue with memory recall.
- **Custom procedural transfer benchmark** — does an agent with Engram do better on tasks it has seen analogues of?

Baselines: flat vector store (Chroma, Pinecone), mem0, Letta/MemGPT, full-context (where feasible).

Each suite should be reproducible from a single command and should not require API keys to *exercise* the pipeline (use cached fixtures); live runs against a real provider should be opt-in.
