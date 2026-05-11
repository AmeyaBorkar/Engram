"""BGE cross-encoder reranker (Tier 1 retrieval-precision uplift).

Wraps `sentence-transformers`' `CrossEncoder` around a BGE reranker
model -- default `BAAI/bge-reranker-v2-m3`. The reranker takes the
top-N candidates from the coarse-to-fine pipeline (typically N=50
when over-fetching) and produces sharper scores than dense bi-
encoder similarity can on its own.

Reranking cost: one cross-encoder forward pass per (query, candidate)
pair. For N=50 candidates and a 200-token candidate, that's ~50
forward passes -- a few hundred ms on CPU, tens of ms on GPU. Stage
6's perf budget at 100k items leaves us ~50x headroom, so the
reranker fits comfortably.

Lazy import: `sentence-transformers` only imports inside the BGE
adapter's `__init__`. Users who don't install the `[reranker]` extra
can still `import engram.retrieve` -- they just can't construct a
`BGEReranker` instance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from engram.retrieve._reranker import RerankCandidate

if TYPE_CHECKING:  # pragma: no cover - import-only for typing
    pass

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class BGEReranker:
    """Cross-encoder reranker backed by `sentence-transformers`.

    Args:
      model: HuggingFace model id (default `BAAI/bge-reranker-v2-m3`).
        Other strong choices: `BAAI/bge-reranker-large`,
        `BAAI/bge-reranker-base`, `mixedbread-ai/mxbai-rerank-large-v1`.
      device: `"cuda"` / `"cpu"` / `None` (auto-detect).
      max_length: token cap per candidate; longer candidates truncate.
        Default 512 fits comfortably in BGE-reranker-v2-m3's context.
      batch_size: cross-encoder forward batch size. CPU defaults to
        16; GPU defaults to 64 for throughput.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - tested via skip
            raise ImportError(
                "BGEReranker requires the `[reranker]` extra "
                "(pip install engram-memory[reranker])"
            ) from exc

        self._model_name = model
        self._max_length = max_length
        self._device = device
        self._batch_size = (
            batch_size if batch_size is not None else self._default_batch_size(device)
        )
        self._model: Any = CrossEncoder(model, max_length=max_length, device=device)

    @staticmethod
    def _default_batch_size(device: str | None) -> int:
        # `cuda` users typically have memory headroom; CPU users
        # benefit from smaller batches that fit in cache.
        if device == "cuda":
            return 64
        return 16

    @property
    def name(self) -> str:
        return f"bge-reranker:{self._model_name}"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, cand.result.content) for cand in candidates]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]


__all__ = ["DEFAULT_MODEL", "BGEReranker"]
