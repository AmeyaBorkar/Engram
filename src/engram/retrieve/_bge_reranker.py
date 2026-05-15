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

Lazy load: the model is constructed on the first `rerank()` call,
not at `__init__`. Constructing a `BGEReranker(...)` is cheap; the
1-2 GB load only happens when you actually rerank. Pre-warm by
calling `rerank(q, [])` if you want the load to happen at a known
time. `unload()` releases the model.

Thread safety: a single `CrossEncoder.predict` call is internally
batched; concurrent `rerank()` calls share the same underlying
model. The torch / numpy paths are not guaranteed re-entrant on the
same tensors, so this class serializes calls behind an internal
lock. Callers that need parallel scoring across multiple cores
should construct one reranker per worker.

Async: `arerank()` hops onto `asyncio.to_thread` so the cross-
encoder forward pass does not block the event loop. The optional
`executor` kwarg lets callers route the work through a shared
`concurrent.futures.Executor` (e.g. a process pool) when the GIL is
the bottleneck.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from concurrent.futures import Executor
from typing import TYPE_CHECKING, Any, Literal

from engram.retrieve._reranker import RerankCandidate

if TYPE_CHECKING:  # pragma: no cover - import-only for typing
    pass

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
        self._batch_size = (
            batch_size if batch_size is not None else self._default_batch_size(device)
        )
        self._dtype: str = _resolve_dtype(dtype, device or "cpu")
        # Lazy state. The model load is deferred to the first call.
        self._model: Any = None
        self._load_lock = threading.Lock()
        # The cross-encoder + its underlying torch / numpy paths are
        # not guaranteed re-entrant on the same tensors; serialize
        # calls behind an internal lock.
        self._predict_lock = threading.Lock()

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

    @property
    def is_loaded(self) -> bool:
        """True once the underlying CrossEncoder has been built."""
        return self._model is not None

    def _ensure_loaded(self) -> None:
        """Load the cross-encoder if it has not been already.

        Thread-safe via `_load_lock`. Idempotent after a successful load.
        """
        if self._model is not None:
            return
        with self._load_lock:
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
                self._model_name,
                max_length=self._max_length,
                device=self._device,
            )
            if self._dtype == "float16":
                # CrossEncoder exposes the underlying HF model at .model.
                model.model.half()
            self._model = model

    def unload(self) -> None:
        """Release the loaded model. Next `rerank()` will re-load."""
        with self._load_lock:
            self._model = None

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[float]:
        if not candidates:
            return []
        self._ensure_loaded()
        pairs = [(query, cand.result.content) for cand in candidates]
        with self._predict_lock:
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
        *,
        executor: Executor | None = None,
    ) -> list[float]:
        """Async wrapper that runs the cross-encoder off the event loop.

        Pass `executor` to dispatch the work through a shared
        `concurrent.futures.Executor` (e.g. a process pool) when the
        GIL is the bottleneck; default `None` uses asyncio's default
        thread pool.
        """
        if not candidates:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.rerank,
            query,
            list(candidates),
        )


__all__ = ["DEFAULT_MODEL", "BGEReranker"]
