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
adapter's `__init__`. Users who don't install the `[reranker]` extra
can still `import engram.retrieve` -- they just can't construct a
`BGEReranker` instance.
"""

from __future__ import annotations

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
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - tested via skip
            raise ImportError(
                "BGEReranker requires the `[reranker]` extra (pip install engram-memory[reranker])"
            ) from exc

        self._model_name = model
        self._max_length = max_length
        self._device = device
        self._batch_size = (
            batch_size if batch_size is not None else self._default_batch_size(device)
        )
        self._model: Any = CrossEncoder(model, max_length=max_length, device=device)
        resolved_dtype = _resolve_dtype(dtype, device or "cpu")
        self._dtype = resolved_dtype
        if resolved_dtype == "float16":
            # CrossEncoder exposes the underlying HF model at .model.
            self._model.model.half()

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
        # Shares the engram._gpu_lock semaphore with LocalEmbedder; in
        # a parallel-bench setup the cross-encoder pass is the bigger
        # forward (rerank pool ~50 with seq~200), so capping concurrency
        # here is what actually keeps VRAM bounded on 12 GB at fp32.
        from engram._gpu_lock import gpu_section

        with gpu_section():
            scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [float(s) for s in scores]


__all__ = ["DEFAULT_MODEL", "BGEReranker"]
