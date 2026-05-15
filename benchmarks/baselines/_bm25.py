"""Minimal BM25 baseline used by the hybrid baseline / smoke suite.

This module used to ship its own pure-Python BM25 — readable, but
asymptotically far slower than Engram's production BM25Index (O(query *
N_docs * doc_len) Python scans inside `term in doc` / `doc.count(term)`
loops vs the engine's vectorized numpy posting-list math).

Running benchmarks against a slow baseline inflates the apparent lead
of Engram's own BM25; that's not fair measurement.  This module now
wraps `engram.retrieve._bm25.BM25Index` and preserves the small public
API (`add`, `topk`, `avgdl`, `__len__`) that downstream callers (the
hybrid baseline + tests) consume.
"""

from __future__ import annotations

from engram.retrieve._bm25 import BM25Index


class BM25:
    """In-memory BM25 index over a list of documents.

    Same public surface as the previous standalone implementation:

    * `add(text) -> int` returns the assigned doc index.
    * `topk(query, k) -> list[(doc_idx, score)]` returns the top-k
      results sorted by score descending.
    * `avgdl` and `len()` expose corpus statistics.

    Internally delegates to the engine's BM25Index, which is vectorized
    and handles posting lists with numpy.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        # The engine's BM25Index accepts k1=0 (no tf saturation); the
        # baseline's previous tighter validation is preserved here so the
        # existing tests still pin the constructor contract.
        if k1 <= 0:
            raise ValueError(f"k1 must be > 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        self._index: BM25Index[int] = BM25Index(k1=k1, b=b)
        self._next_idx: int = 0
        self._total_len: int = 0
        # Track lengths locally so `avgdl` works without reaching into
        # the engine's frozen internals.  Tokenization matches the
        # engine's BM25Index (lowercased word-character tokens).
        from engram.retrieve._bm25 import tokenize as _tok

        self._tokenize = _tok

    @property
    def avgdl(self) -> float:
        n = self._next_idx
        return self._total_len / n if n else 0.0

    def __len__(self) -> int:
        return self._next_idx

    def add(self, text: str) -> int:
        """Index `text`, returning the document index assigned."""
        idx = self._next_idx
        self._index.add_doc(idx, text)
        self._next_idx += 1
        self._total_len += len(self._tokenize(text))
        return idx

    def topk(self, query: str, k: int) -> list[tuple[int, float]]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if self._next_idx == 0:
            return []
        return list(self._index.search(query, k=k))
