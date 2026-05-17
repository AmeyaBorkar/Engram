"""BM25 lexical scoring for hybrid retrieval.

Dense vector retrieval (cosine over embeddings) and BM25 lexical
retrieval have complementary failure modes. Dense wins on paraphrase
("my graduation degree" vs "the BBA I earned in 2018"); BM25 wins on
exact tokens that the embedder smooths away (dates, codes, names,
years). Fusing the two via Reciprocal Rank Fusion recovers both -- the
LongMemEval temporal-reasoning and knowledge-update categories
especially benefit from anchoring on the literal year / month string.

The index lives in memory and is built lazily on the first `search()`
call. The storage-level VectorIndex mirrors this pattern; the design
point is that the BM25 cost only shows up when a caller actually opts
into hybrid retrieval via `RetrieveParams.bm25_weight > 0`.

Algorithm: Lucene-style Robertson/Sparck Jones BM25 with default
`k1 = 1.5`, `b = 0.75`. Tokenization is intentionally boring --
lowercase + word characters, no stemming, no stop-list. The embedder
already smooths morphology; BM25's job here is to keep the literal
tokens visible.

Scoring path: every per-term posting list is materialized as a pair of
numpy arrays (doc indices, term frequencies) when the index is frozen,
along with a pre-computed (1 - b + b * doc_len/avgdl) length-norm
vector. `search()` becomes one fancy-indexed scatter-add per query
term -- a constant-factor faster than the Python loop for any corpus
above a few dozen docs.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

DocId = TypeVar("DocId", bound=Hashable)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case the input and split on word boundaries.

    No stop-word removal, no stemming. The embedder side handles those
    forms of normalization; BM25's role in the hybrid is to keep the
    literal tokens visible.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass(slots=True)
class _Posting:
    """One term-in-doc posting: which doc, how many times."""

    doc_idx: int
    tf: int


class BM25Index(Generic[DocId]):
    """In-memory BM25 inverted index over a fixed corpus.

    Construct with `add_doc(id, text)` calls, then `search(query, k)`.
    The index does not support deletion or updates -- rebuild from
    scratch when the corpus changes. For LongMemEval haystacks (one
    fresh corpus per question, ~500 docs) the rebuild is sub-millisecond.

    Empty corpus is a valid state; `search` returns `[]`.

    `k1`, `b`: classic BM25 hyperparameters. Defaults (1.5, 0.75) match
    Lucene's. `k1 in [1.2, 2.0]` and `b in [0.5, 0.9]` are the sane
    ranges; lower `b` reduces length normalization (favors longer
    docs), higher `k1` lets repeated terms keep accumulating mass.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        self._k1 = k1
        self._b = b
        self._ids: list[DocId] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[_Posting]] = {}
        self._avgdl: float = 0.0
        self._frozen = False
        # Populated lazily on the first `search()` from `_freeze()`.
        # All arrays use float32 (small) and int32 (doc indices).
        self._length_norms: np.ndarray | None = None
        self._term_idf: dict[str, float] = {}
        self._term_doc_idx: dict[str, np.ndarray] = {}
        self._term_tf: dict[str, np.ndarray] = {}

    def add_doc(self, doc_id: DocId, text: str) -> None:
        """Add a document to the index. Must be called before any `search`."""
        if self._frozen:
            raise RuntimeError("BM25Index is frozen after the first search(); rebuild from scratch")
        tokens = tokenize(text)
        doc_idx = len(self._ids)
        self._ids.append(doc_id)
        self._lengths.append(len(tokens))
        counts = Counter(tokens)
        for term, tf in counts.items():
            self._postings.setdefault(term, []).append(_Posting(doc_idx, tf))

    def _freeze(self) -> None:
        """Materialize the postings into numpy arrays for fast scoring.

        Idempotent. Called from `search()` on the first invocation. The
        Python-side dict of `_Posting` objects stays around for
        debugging but `search()` never touches it again -- all scoring
        runs through the parallel numpy arrays.
        """
        if self._frozen:
            return
        self._frozen = True
        n = len(self._lengths)
        if n == 0:
            self._avgdl = 0.0
            self._length_norms = np.zeros(0, dtype=np.float32)
            return
        lengths_np = np.asarray(self._lengths, dtype=np.float32)
        avgdl = float(lengths_np.mean()) if lengths_np.size > 0 else 0.0
        self._avgdl = avgdl
        if avgdl > 0:
            self._length_norms = (1.0 - self._b) + self._b * (lengths_np / avgdl)
        else:
            self._length_norms = np.ones(n, dtype=np.float32)
        for term, postings in self._postings.items():
            df = len(postings)
            self._term_idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            self._term_doc_idx[term] = np.asarray([p.doc_idx for p in postings], dtype=np.int64)
            self._term_tf[term] = np.asarray([p.tf for p in postings], dtype=np.float32)

    def search(self, query: str, k: int) -> list[tuple[DocId, float]]:
        """Return top-`k` `(doc_id, bm25_score)` pairs, score descending.

        Empty corpus or empty query -> empty list. Ties in score break
        by insertion order (stable across runs).
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not self._ids:
            return []
        self._freeze()
        assert self._length_norms is not None
        terms = tokenize(query)
        if not terms:
            return []
        n = len(self._ids)
        scores = np.zeros(n, dtype=np.float32)
        k1_plus_one = self._k1 + 1.0
        # One scatter-add per unique query term. Each posting list has
        # at most one entry per (term, doc) so the doc-index arrays are
        # already unique and fancy-indexed `scores[doc_idx] +=` is safe
        # (no aliasing).
        for term in set(terms):
            idf = self._term_idf.get(term)
            if idf is None:
                continue
            doc_idx = self._term_doc_idx[term]
            tf = self._term_tf[term]
            norm = self._length_norms[doc_idx]
            tf_score = tf * k1_plus_one / (tf + self._k1 * norm)
            scores[doc_idx] += idf * tf_score
        pos = scores > 0
        if not pos.any():
            return []
        pos_idx = np.nonzero(pos)[0]
        if pos_idx.size <= k:
            order = pos_idx[np.argsort(-scores[pos_idx], kind="stable")]
        else:
            top_local = np.argpartition(-scores[pos_idx], k - 1)[:k]
            order_local = top_local[np.argsort(-scores[pos_idx[top_local]], kind="stable")]
            order = pos_idx[order_local]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in order[:k]]

    def __len__(self) -> int:
        return len(self._ids)


def reciprocal_rank_fusion(
    rankings: list[list[tuple[DocId, float]]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[DocId, float]]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    `RRF(d) = sum_r w_r / (k + rank_r(d))` over the rankings that
    contain `d`.  `w_r` is the per-ranking weight (defaults to 1.0
    when `weights` is None) — callers that want to scale BM25 vs dense
    independently pass a `weights` tuple shaped like `rankings`.

    The `k = 60` smoothing constant is the standard value from the RRF
    paper.  Output is `(doc_id, fused_score)` sorted by fused score
    descending; score sign is preserved (higher is better).

    Duplicate doc_ids within a single ranking only count once — the
    inner per-ranking dedup mirrors the standard RRF formulation and
    prevents a buggy upstream that emits the same id twice in one
    stream from double-weighting it.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError(
            f"weights length {len(weights)} does not match rankings length {len(rankings)}"
        )
    if weights is not None and any(w < 0 for w in weights):
        raise ValueError(f"weights must be >= 0, got {list(weights)!r}")
    fused: dict[DocId, float] = {}
    for i, ranking in enumerate(rankings):
        w = float(weights[i]) if weights is not None else 1.0
        if w <= 0.0:
            continue
        seen: set[DocId] = set()
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
