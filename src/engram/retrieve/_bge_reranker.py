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

`dtype="auto"` (default) loads in fp16 on CUDA, fp32 on CPU. fp16
halves the model VRAM footprint (BGE-reranker-v2-m3: ~2.3 GB fp32 ->
~1.1 GB fp16) so a 12 GB GPU can host the reranker alongside a
1.5 B-param embedder (stella) without OOM.

Lazy import: `sentence-transformers` only imports inside the BGE
adapter's `_ensure_loaded`. Users who don't install the `[reranker]`
extra can still `import engram.retrieve` AND construct a
`BGEReranker` instance -- the model only loads on the first `rerank`
call. Lets a caller wire up `Memory(retrieve_params=...,
reranker=BGEReranker(...))` without paying the 1-2 GB VRAM cost
until the first real retrieve.

Score scale: BGE-reranker-v2-m3 emits raw cross-encoder logits,
typically in the range [-10, +10]. These are NOT in the [0, 1] band
the dense cosine path produces, so `RetrievalResult.confidence`
(which is documented as [0, 1]) must NOT be derived from a reranker
score. The engine instead preserves the dense cosine via
`_Candidate.confidence_score`; the reranker score lives only in the
ordering signal `RetrievalResult.score`. See `_engine._finalize` for
the split.

Thread safety: `CrossEncoder.predict` calls into PyTorch, which is
NOT thread-safe for shared models on a single device. Two threads
each calling `rerank` on the same `BGEReranker` instance race on the
model's internal buffers. Use a process-pool executor for parallel
rerank work, OR construct one `BGEReranker` per worker thread, OR
serialize calls via a per-instance lock at the caller layer. The
`arerank` async wrapper offloads the work to the default thread pool
which is the typical "one model, many requests" path; it relies on
the caller ensuring at most one outstanding call per instance.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal

from engram.retrieve._reranker import RerankCandidate

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

DType = Literal["auto", "float16", "float32"]


def _resolve_dtype(requested: DType, actual_device: str) -> str:
    if requested == "float16":
        return "float16"
    if requested == "float32":
        return "float32"
    if actual_device.lower().startswith("cuda"):
        return "float16"
    return "float32"


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
      dtype: `"auto"` (default) loads fp16 on CUDA, fp32 elsewhere.
        Force `"float32"` if you need bitwise reproducibility against
        a previously-recorded baseline.

    Lazy loading: the underlying HuggingFace model is NOT downloaded
    or loaded into memory in `__init__`. The first `rerank()` call
    triggers `_ensure_loaded()`. Construction is cheap (a few
    microseconds) so callers can hold a `BGEReranker` reference in
    config without paying the VRAM/RAM cost up front.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int | None = None,
        dtype: DType = "auto",
    ) -> None:
        self._model_name = model
        self._max_length = max_length
        self._device = device
        self._dtype: DType = dtype
        self._batch_size = (
            batch_size if batch_size is not None else self._default_batch_size(device)
        )
        # `_model` stays None until the first `rerank()` call. The
        # heavy import + model download happens in `_ensure_loaded`.
        self._model: Any | None = None
        self._resolved_dtype: str | None = None

    def _ensure_loaded(self) -> None:
        """Load the cross-encoder if it isn't already loaded.

        Called by `rerank()` lazily. Safe to call repeatedly -- the
        `_model is None` check makes it a no-op after the first
        successful load. NOT thread-safe: two simultaneous first
        calls from different threads will both trigger the load. The
        caller (`HierarchicalRetriever._finalize`) is single-threaded
        per retrieve, so the race is only theoretical.
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - tested via skip
            raise ImportError(
                "BGEReranker requires the `[reranker]` extra "
                "(pip install engram-memory[reranker])"
            ) from exc
        model = CrossEncoder(
            self._model_name, max_length=self._max_length, device=self._device
        )
        resolved_dtype = _resolve_dtype(self._dtype, self._device or "cpu")
        if resolved_dtype == "float16":
            # CrossEncoder exposes the underlying HF model at .model.
            model.model.half()
        self._model = model
        self._resolved_dtype = resolved_dtype

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
        self._ensure_loaded()
        assert self._model is not None
        pairs = [(query, cand.result.content) for cand in candidates]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]

    async def arerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[float]:
        """Async wrapper that runs `rerank` in the default thread pool.

        Lets an `await memory.aretrieve(...)` path keep the event loop
        free during the cross-encoder forward pass. NOT a parallelism
        primitive: see the module docstring about thread safety. The
        caller must ensure at most one outstanding `arerank` per
        `BGEReranker` instance (run multiple instances if you want
        parallelism, or batch-size-stack inside one call).
        """
        return await asyncio.to_thread(self.rerank, query, candidates)


__all__ = ["DEFAULT_MODEL", "BGEReranker"]
