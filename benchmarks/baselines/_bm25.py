"""Minimal BM25 implementation used by the hybrid baseline.

Self-contained — no `rank_bm25` dependency. Lowercase whitespace tokenizer;
`k1=1.5`, `b=0.75` per common defaults. Scores are computed on demand
against the full corpus, which is fine at smoke-benchmark scale.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """In-memory BM25 index over a list of documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError(f"k1 must be > 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        self._k1 = k1
        self._b = b
        self._docs: list[list[str]] = []
        self._doc_freqs: Counter[str] = Counter()
        self._total_len = 0

    @property
    def avgdl(self) -> float:
        n = len(self._docs)
        return self._total_len / n if n else 0.0

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, text: str) -> int:
        """Index `text`, returning the document index assigned."""
        tokens = _tokenize(text)
        for term in set(tokens):
            self._doc_freqs[term] += 1
        self._docs.append(tokens)
        self._total_len += len(tokens)
        return len(self._docs) - 1

    def topk(self, query: str, k: int) -> list[tuple[int, float]]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not self._docs:
            return []
        scores = self._score_all(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def _score_all(self, query_tokens: Sequence[str]) -> list[float]:
        n_docs = len(self._docs)
        avgdl = self.avgdl or 1.0  # guard division by zero
        scores = [0.0] * n_docs
        for term in query_tokens:
            df = self._doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            for i, doc in enumerate(self._docs):
                if term not in doc:
                    continue
                tf = doc.count(term)
                doc_len = len(doc)
                num = tf * (self._k1 + 1)
                den = tf + self._k1 * (1 - self._b + self._b * doc_len / avgdl)
                scores[i] += idf * num / den
        return scores
