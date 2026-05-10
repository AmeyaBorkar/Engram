# Baselines

Reserved for adapters to the systems we compare Engram against. Each adapter is a thin shim that exposes a system's `observe` / `retrieve` (or equivalent) under a common interface so the harness can drive it the same way it drives Engram.

Tracked baselines (full list with rationale: `../SOTA.md`):

- Chroma — flat dense vector store.
- Chroma + BM25 — hybrid dense+sparse.
- Letta / MemGPT — paged hierarchical memory.
- Zep / Graphiti — knowledge-graph memory.
- Cognee — alternative graph-RAG approach.
- HippoRAG — PageRank-based multi-hop retrieval.
- mem0 — summarization-based memory.
- A-MEM — Zettelkasten-style linked notes.
- Full-context (Sonnet / Opus, 1M) — upper-bound; not memory but a useful ceiling.

Adapters install via their own extras (e.g. `pip install -e "../[baselines-zep]"`) so the core library's dependency graph stays clean.
