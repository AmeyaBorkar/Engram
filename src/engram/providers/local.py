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

Asymmetric models (stella, e5, ...) use different prompts/prefixes for
queries vs passages. The `_QUERY_ENCODE_KWARGS` table maps known HF ids
to the sentence-transformers encode kwargs that should be applied when
the call is encoding a search QUERY. `embed_query()` honors the table;
the document-side `embed()` ignores it. Models not in the table fall
through with no special handling -- correct for bge-large where the
optional query prompt was not used in the v0.1.0 receipt and we keep
that path bit-identical.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is not installed. Install with: "
        "pip install 'engram[bench]'   # or: pip install sentence-transformers"
    ) from _exc

from engram.providers._cache import Cache, content_hash

_LOG = logging.getLogger("engram.providers.local")

DType = Literal["auto", "float16", "float32"]

# Per-model sentence-transformers encode kwargs to apply when the input
# is a search QUERY (not a passage / document being stored). Empty /
# missing entry means "no special handling": symmetric models like
# bge-large-en-v1.5 fall through this way. Stella requires the
# `s2p_query` prompt that ships in its model config; e5 needs a literal
# "query: " prefix because its prompts are documented as raw prefixes.
_QUERY_ENCODE_KWARGS: dict[str, dict[str, str]] = {
    "dunzhang/stella_en_1.5B_v5": {"prompt_name": "s2p_query"},
    "intfloat/e5-large-v2": {"prompt": "query: "},
    "intfloat/e5-base-v2": {"prompt": "query: "},
    "intfloat/e5-small-v2": {"prompt": "query: "},
    "intfloat/e5-mistral-7b-instruct": {"prompt": "query: "},
}


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

    `cache_size` enables an in-memory LRU keyed on (model, dtype, text).
    On benchmarks where the same question / template / haystack turn is
    embedded multiple times across the verify / iterative-react / multi-
    query paths, the cache turns N GPU encodes into N-k cache reads.
    Default 4096 entries; pass 0 to disable.
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
        cache_size: int = 4096,
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
            # `.half()` mutates the model in place.  If it raises mid-way
            # (e.g. on an unsupported model architecture, or an MPS
            # device that doesn't support fp16 for some op), we'd leave
            # the instance carrying a partially-converted SentenceTrans-
            # former.  Drop the reference and re-raise so the caller
            # gets a clean exception instead of a half-initialized
            # LocalEmbedder that fails opaquely on first encode().
            try:
                self._st.half()
            except Exception:
                self._st = None  # type: ignore[assignment]
                raise
        native_dim = int(self._st.get_sentence_embedding_dimension() or 0)
        if dim is not None and dim != native_dim:
            raise ValueError(f"requested dim={dim} does not match model native dim={native_dim}")
        self.dim = native_dim
        self._query_kwargs: dict[str, str] = _QUERY_ENCODE_KWARGS.get(model, {})
        self._cache: Cache[list[float]] | None = (
            Cache[list[float]](max_size=cache_size) if cache_size > 0 else None
        )
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
                "LocalEmbedder ready: model=%s device=%s dim=%d batch=%d dtype=%s "
                "asymmetric_query=%s cache=%d",
                model,
                chosen_device,
                native_dim,
                batch_size,
                resolved_dtype,
                bool(self._query_kwargs),
                cache_size,
            )

    def _cache_key(self, text: str, *, kind: str) -> str:
        # `kind` separates query vs document encodings -- the same text
        # encoded as a query (s2p_query prompt) and as a document (no
        # prompt) gives DIFFERENT vectors on asymmetric models, so they
        # must not share a cache slot.
        return content_hash(self.model, self._dtype, kind, text)

    def _encode(self, texts: list[str], *, extra: dict[str, Any] | None = None) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "batch_size": self._batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": self._normalize,
            "show_progress_bar": False,
        }
        if extra:
            kwargs.update(extra)
        vectors = self._st.encode(texts, **kwargs)
        return [row.tolist() for row in vectors]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        text_list = list(texts)
        if self._cache is None:
            return self._encode(text_list)
        # Partition into cache hits + misses; encode only the misses.
        results: list[list[float] | None] = [None] * len(text_list)
        miss_idx: list[int] = []
        miss_text: list[str] = []
        for i, t in enumerate(text_list):
            cached = self._cache.get(self._cache_key(t, kind="doc"))
            if cached is not None:
                results[i] = cached
            else:
                miss_idx.append(i)
                miss_text.append(t)
        if miss_text:
            new_vecs = self._encode(miss_text)
            for idx, vec in zip(miss_idx, new_vecs, strict=True):
                results[idx] = vec
                self._cache.set(self._cache_key(text_list[idx], kind="doc"), vec)
        # Every slot is filled by construction.
        return [r for r in results if r is not None]

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query, applying asymmetric prompts when the
        model needs them.

        Symmetric models (bge-large, mxbai, ...) fall through to a
        regular encode -- the result is bit-identical to `embed([q])[0]`,
        so the hierarchy retriever can safely prefer this method when
        present without changing behavior on symmetric models.
        """
        if self._cache is not None:
            cached = self._cache.get(self._cache_key(query, kind="query"))
            if cached is not None:
                return cached
        extra = dict(self._query_kwargs) if self._query_kwargs else None
        vec = self._encode([query], extra=extra)[0]
        if self._cache is not None:
            self._cache.set(self._cache_key(query, kind="query"), vec)
        return vec

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        import asyncio

        return await asyncio.to_thread(self.embed, list(texts))

    async def aembed_query(self, query: str) -> list[float]:
        import asyncio

        return await asyncio.to_thread(self.embed_query, query)

    @property
    def cache(self) -> Cache[list[float]] | None:
        """The underlying LRU cache, or None if caching is disabled.

        Exposed for tests / observability -- callers can read
        `.hit_rate`, `.hits`, `.misses` to track behavior.
        """
        return self._cache

    def manifest_hash(self) -> str:
        norm = "norm" if self._normalize else "raw"
        # Device + dtype flow into the hash so a CPU run, a CUDA-fp16
        # run, and a CUDA-fp32 run of the same model land in distinct
        # manifest rows -- helpful when the determinism floor between
        # the three differs (mixed precision, cuBLAS reductions, ...).
        # Asymmetric-query state goes into the hash too so a model
        # encoded with `s2p_query` prompts doesn't share a manifest row
        # with the same model encoded symmetrically.
        async_flag = "asym" if self._query_kwargs else "sym"
        return (
            f"local-embed/{self.model}/dim={self.dim}/{norm}/"
            f"device={self._device}/dtype={self._dtype}/{async_flag}/v3"
        )
