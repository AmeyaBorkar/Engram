"""Local sentence-transformers embedder.

Behind the `[bench]` extra (or any environment with `sentence-transformers`
installed). The point is: real-quality embeddings without an API key,
running on the user's GPU or CPU.

Defaults to `BAAI/bge-large-en-v1.5` -- 330 M params, 1024-dim, top-tier
on MTEB retrieval (~54). Override with any sentence-transformers model
id. `RECOMMENDED_MODELS` enumerates the top choices.

GPU auto-detect: `torch.cuda.is_available()` chooses CUDA; CPU falls
out otherwise. Override explicitly via `device="cuda:1"` etc. Embeddings
are L2-normalized at encode time so cosine similarity reduces to a
dot product in storage.

First call to `embed` triggers model download into the HuggingFace
cache (default `~/.cache/huggingface/hub/`). For `BAAI/bge-large-en-v1.5`
that's roughly 1.3 GiB on disk; the download happens once.

`dtype="auto"` (default) loads in fp16 when CUDA is the resolved
device, fp32 otherwise. fp16 halves the VRAM footprint and the
embedding noise is negligible after L2-normalization + cross-encoder
rerank. Override via `dtype="float32"` if you want full precision on
GPU (e.g. for determinism / regression-locking a numeric baseline).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is not installed. Install with: "
        "pip install 'engram[bench]'   # or: pip install sentence-transformers"
    ) from _exc

_LOG = logging.getLogger("engram.providers.local")

DType = Literal["auto", "float16", "float32"]


def _resolve_dtype(requested: DType, actual_device: str) -> str:
    """Pick the concrete dtype, given the user request + actual device.

    `auto` -> fp16 on CUDA, fp32 elsewhere. CPU fp16 is supported but
    rarely worth the precision loss since CPU is already the bottleneck.
    """
    if requested == "float16":
        return "float16"
    if requested == "float32":
        return "float32"
    # auto:
    if actual_device.lower().startswith("cuda"):
        return "float16"
    return "float32"


# Top embedders ranked roughly by MTEB retrieval score. Pick by the
# tier that fits your GPU memory + latency budget. Each entry is
# (HF_id, dim, approximate_params, notes). The dim is fixed; some
# matryoshka models support truncation, but we use the native dim.
RECOMMENDED_MODELS: dict[str, dict[str, object]] = {
    # ---- Tier S: state of the art -----------------------------------
    "nvidia/NV-Embed-v2": {
        "dim": 4096,
        "params": "7.85B",
        "notes": (
            "MTEB ~69 retrieval, top of the leaderboard as of 2024-09. "
            "Heavy: ~16 GB VRAM in fp16. CPU is impractical."
        ),
    },
    "dunzhang/stella_en_1.5B_v5": {
        "dim": 8192,
        "params": "1.5B",
        "notes": (
            "MTEB ~66 retrieval; great quality-per-VRAM. Matryoshka-"
            "trained -- truncating to 1024-dim still scores strongly."
        ),
    },
    # ---- Tier A: production workhorses ------------------------------
    "BAAI/bge-large-en-v1.5": {
        "dim": 1024,
        "params": "330M",
        "notes": (
            "Default. MTEB ~54 retrieval, ~1.3 GiB on disk, fits in 4 GB "
            "VRAM at fp16. The Stage 6 LongMemEval-S 71.4% receipt used "
            "this."
        ),
    },
    "mixedbread-ai/mxbai-embed-large-v1": {
        "dim": 1024,
        "params": "330M",
        "notes": "MTEB ~54.4 retrieval. Drop-in alternative to bge-large.",
    },
    "BAAI/bge-m3": {
        "dim": 1024,
        "params": "568M",
        "notes": (
            "Multilingual; dense+sparse+colbert in one model. Use when "
            "the corpus isn't predominantly English."
        ),
    },
    # ---- Tier B: smaller / faster -----------------------------------
    "BAAI/bge-base-en-v1.5": {
        "dim": 768,
        "params": "110M",
        "notes": "Faster, lower-quality variant of bge-large.",
    },
    "intfloat/e5-large-v2": {
        "dim": 1024,
        "params": "335M",
        "notes": "Strong alt to bge-large; needs 'query:'/'passage:' prefixes.",
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "dim": 768,
        "params": "137M",
        "notes": "Matryoshka-trained; can truncate to 64-768 dims.",
    },
}


def _detect_device() -> str:
    """Best-effort device pick. CUDA > MPS > CPU."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch ships with sentence-transformers
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalEmbedder:
    """Sentence-transformers wrapped to satisfy `EmbeddingProvider`.

    The class loads the model into memory on construction, so two
    `LocalEmbedder` instances pointed at the same model share nothing
    -- create once, embed many. The bench harness builds exactly one
    per run.
    """

    name: str = "local-embed"

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        *,
        device: str | None = None,
        batch_size: int = 64,
        dim: int | None = None,
        normalize: bool = True,
        dtype: DType = "auto",
    ) -> None:
        chosen_device = device if device is not None else _detect_device()
        self._st = _SentenceTransformer(model, device=chosen_device)
        self.model = model
        self._device = chosen_device
        self._batch_size = batch_size
        self._normalize = normalize
        resolved_dtype = _resolve_dtype(dtype, chosen_device)
        self._dtype = resolved_dtype
        if resolved_dtype == "float16":
            self._st.half()
        native_dim = int(self._st.get_sentence_embedding_dimension() or 0)
        if dim is not None and dim != native_dim:
            raise ValueError(f"requested dim={dim} does not match model native dim={native_dim}")
        self.dim = native_dim
        if chosen_device == "cpu":
            _LOG.warning(
                "LocalEmbedder running on CPU (model=%s, dim=%d). Expect ~50x slower "
                "than CUDA. If you have an NVIDIA GPU, install CUDA-enabled torch via "
                "`pip install torch --index-url https://download.pytorch.org/whl/cu124`.",
                model,
                native_dim,
            )
        else:
            _LOG.info(
                "LocalEmbedder ready: model=%s device=%s dim=%d batch=%d dtype=%s",
                model,
                chosen_device,
                native_dim,
                batch_size,
                resolved_dtype,
            )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._st.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        # sentence-transformers returns numpy array of shape (n, d).
        return [row.tolist() for row in vectors]

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        import asyncio

        return await asyncio.to_thread(self.embed, list(texts))

    def manifest_hash(self) -> str:
        norm = "norm" if self._normalize else "raw"
        # Device + dtype flow into the hash so a CPU run, a CUDA-fp16
        # run, and a CUDA-fp32 run of the same model land in distinct
        # manifest rows -- helpful when the determinism floor between
        # the three differs (mixed precision, cuBLAS reductions, ...).
        return (
            f"local-embed/{self.model}/dim={self.dim}/{norm}/"
            f"device={self._device}/dtype={self._dtype}/v2"
        )
