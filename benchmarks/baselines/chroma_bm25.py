"""Chroma + BM25 hybrid baseline (Reciprocal Rank Fusion).

Hybrid retrieval almost always beats pure dense (per `benchmarks/SOTA.md`),
so the bench harness measures it explicitly and Engram has to clear that
bar — not just the flat dense bar — to claim a SOTA win.

Fusion is RRF with `K=60`: per the canonical formulation,

    score(d) = sum over rankers r:  1 / (K + rank_r(d))

where `rank_r(d)` is the 1-indexed rank of `d` in ranker `r`'s top-k.
RRF requires no score-distribution alignment between rankers (cosine
similarities and BM25 scores live on entirely different scales) and is
empirically competitive with weighted linear combinations.
"""

from __future__ import annotations

import uuid

from benchmarks.baselines._bm25 import BM25
from benchmarks.baselines.chroma import ChromaRetriever
from engram.bench import Hit
from engram.providers import EmbeddingProvider

_RRF_K = 60


class ChromaBM25Retriever:
    """Dense (Chroma) + sparse (BM25) hybrid with Reciprocal Rank Fusion."""

    name: str = "chroma+bm25"

    def __init__(self, embedder: EmbeddingProvider | None = None) -> None:
        self._chroma = ChromaRetriever(embedder=embedder)
        self._bm25 = BM25()
        self._docs: dict[str, str] = {}
        self._bm25_idx_to_id: dict[int, str] = {}

    def add(self, content: str, doc_id: str | None = None) -> str:
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        self._chroma.add(content, doc_id=doc_id)
        bm25_idx = self._bm25.add(content)
        self._bm25_idx_to_id[bm25_idx] = doc_id
        self._docs[doc_id] = content
        return doc_id

    def query(self, query: str, k: int) -> list[Hit]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not self._docs:
            return []

        # Pull the top-k from each ranker; over-pulling slightly helps fusion
        # rescue items that one ranker missed entirely.
        topk = max(k * 2, 10)
        dense = self._chroma.query(query, k=min(topk, len(self._docs)))
        sparse = self._bm25.topk(query, k=min(topk, len(self._docs)))

        rrf: dict[str, float] = {}
        for rank, hit in enumerate(dense):
            rrf[hit.id] = rrf.get(hit.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        for rank, (idx, _score) in enumerate(sparse):
            doc_id = self._bm25_idx_to_id[idx]
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)

        ranked = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(id=doc_id, content=self._docs[doc_id], score=score) for doc_id, score in ranked]
