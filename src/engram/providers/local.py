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

Lazy loading: the underlying `SentenceTransformer` is built on the
first encode call, not at construction. Importing this module is
cheap; the multi-GB model load only happens when you actually need
to embed. Pre-warm by calling `.embed([])` or `.embed_query("")` if
you want the load to happen at a known time. `unload()` releases the
loaded model so a long-running process can free GPU memory between
phases. `dim` must be discoverable without loading -- when not passed
explicitly, the constructor consults `RECOMMENDED_MODELS` first and
falls back to an eager load when the model is unknown.

First call to `embed` triggers model download into the HuggingFace
cache (default `~/.cache/huggingface/hub/`). For `BAAI/bge-large-en-v1.5`
that's roughly 1.3 GiB on disk; the download happens once.

`dtype="auto"` (default) loads in fp16 when CUDA is the resolved
device, fp32 otherwise. fp16 halves the VRAM footprint and the
embedding noise is negligible after L2-normalization + cross-encoder
rerank. Override via `dtype="float32"` if you want full precision on
GPU (e.g. for determinism / regression-locking a numeric baseline).

Matryoshka truncation: pass `target_dim` (must be <= native_dim) to
truncate the final embedding to a shorter vector and re-normalize.
This works on any model but is only quality-preserving for Matryoshka-
trained models (Stella, nomic-embed-text-v1.5). On non-Matryoshka
models truncation is allowed but emits a warning.

Asymmetric models (stella, e5, ...) use different prompts/prefixes for
queries vs passages. The `_QUERY_ENCODE_KWARGS` table maps known HF ids
to the sentence-transformers encode kwargs that should be applied when
the call is encoding a search QUERY. `embed_query()` honors the table;
the document-side `embed()` ignores it. Models not in the table fall
through with no special handling -- on symmetric models the query and
document cache slots collapse together so warming one warms the other.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from typing import Any, Literal

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


# Matryoshka-trained models: truncating to a smaller dim and re-
# normalizing is documented as quality-preserving on these. Anything
# else gets a warning when target_dim < native_dim.
_MATRYOSHKA_MODELS: set[str] = {
    "dunzhang/stella_en_1.5B_v5",
    "nomic-ai/nomic-embed-text-v1.5",
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


def _catalog_dim(model: str) -> int | None:
    """Native dim from `RECOMMENDED_MODELS` if listed, else None."""
    entry = RECOMMENDED_MODELS.get(model)
    if entry is None:
        return None
    dim = entry.get("dim")
    return int(dim) if isinstance(dim, int) else None


class LocalEmbedder:
    """Sentence-transformers wrapped to satisfy `EmbeddingProvider`.

    The model is loaded on the first encode call, not at construction.
    `dim` is settled at construction either from the caller, the
    `RECOMMENDED_MODELS` catalog, or an eager load if neither is
    available. Two `LocalEmbedder` instances pointed at the same model
    share nothing -- create once, embed many.

    `cache_size` enables an in-memory LRU keyed on (model, dtype, text).
    On benchmarks where the same question / template / haystack turn is
    embedded multiple times across the verify / iterative-react / multi-
    query paths, the cache turns N GPU encodes into N-k cache reads.
    Default 4096 entries; pass 0 to disable.

    `target_dim` enables Matryoshka-style truncation: when set <= the
    native dim, every output vector is truncated to that length and
    re-normalized (when `normalize=True`). Only quality-preserving on
    Matryoshka-trained models; non-Matryoshka models still allow the
    truncation but emit a single warning.

    `unload()` releases the loaded model (and the in-memory cache).
    Re-using the instance after `unload()` triggers a re-load.
    """

    name: str = "local-embed"

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        *,
        device: str | None = None,
        batch_size: int = 64,
        dim: int | None = None,
        target_dim: int | None = None,
        normalize: bool = True,
        dtype: DType = "auto",
        cache_size: int = 4096,
    ) -> None:
        self.model = model
        chosen_device = device if device is not None else _detect_device()
        self._device = chosen_device
        self._batch_size = batch_size
        self._normalize = normalize
        resolved_dtype = _resolve_dtype(dtype, chosen_device)
        self._dtype = resolved_dtype
        self._query_kwargs: dict[str, str] = _QUERY_ENCODE_KWARGS.get(model, {})
        self._cache_size = cache_size
        self._cache: Cache[list[float]] | None = (
            Cache[list[float]](max_size=cache_size) if cache_size > 0 else None
        )
        # Lazy state: the model loads on first encode. Use a lock so two
        # async/threaded callers cannot kick off duplicate loads.
        self._st: Any = None
        self._load_lock = threading.Lock()

        # Settle `dim` without forcing the load:
        #   (1) caller passed it: trust them.
        #   (2) catalog knows: use that.
        #   (3) eager-load to discover (rare; unknown model with no dim hint).
        native_dim = _catalog_dim(model)
        if dim is None and native_dim is None:
            # Forced eager load: there is no way to know the dim without
            # actually loading the model.
            self._ensure_loaded()
            native_dim = int(self._st.get_sentence_embedding_dimension() or 0)
        elif dim is not None:
            native_dim = dim
        # At this point native_dim is set.
        assert native_dim is not None and native_dim > 0
        self._native_dim = native_dim

        # Matryoshka truncation: pin target_dim, validate bounds.
        if target_dim is not None:
            if target_dim < 1 or target_dim > native_dim:
                raise ValueError(
                    f"target_dim={target_dim} must be in [1, native_dim={native_dim}]"
                )
            if target_dim != native_dim and model not in _MATRYOSHKA_MODELS:
                _LOG.warning(
                    "LocalEmbedder: target_dim=%d on non-Matryoshka model %s; "
                    "truncation will work but quality is not guaranteed.",
                    target_dim,
                    model,
                )
        self._target_dim = target_dim
        self.dim = target_dim if target_dim is not None else native_dim

        if chosen_device == "cpu":
            _LOG.warning(
                "LocalEmbedder configured for CPU (model=%s, dim=%d). Expect "
                "~50x slower than CUDA. If you have an NVIDIA GPU, install "
                "CUDA-enabled torch via `pip install torch --index-url "
                "https://download.pytorch.org/whl/cu124`.",
                model,
                self.dim,
            )

    # --- lazy model loading -------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the SentenceTransformer if it has not been already.

        Thread-safe via `_load_lock`. Idempotent after a successful load.
        """
        if self._st is not None:
            return
        with self._load_lock:
            if self._st is not None:
                return
            try:
                from sentence_transformers import (
                    SentenceTransformer as _SentenceTransformer,
                )
            except ImportError as exc:  # pragma: no cover - install-time path
                raise ImportError(
                    "sentence-transformers is not installed. Install with: "
                    "pip install 'engram[bench]'   # or: pip install sentence-transformers"
                ) from exc
            st = _SentenceTransformer(self.model, device=self._device)
            if self._dtype == "float16":
                st.half()
            _LOG.info(
                "LocalEmbedder loaded: model=%s device=%s native_dim=%d "
                "target_dim=%s batch=%d dtype=%s asymmetric_query=%s cache=%d",
                self.model,
                self._device,
                self._native_dim,
                self._target_dim,
                self._batch_size,
                self._dtype,
                bool(self._query_kwargs),
                self._cache_size,
            )
            self._st = st

    def unload(self) -> None:
        """Release the loaded model + clear the in-memory cache.

        Useful in long-running processes that swap models between
        phases (e.g. embedder vs. reranker on the same GPU). After
        unload, the next encode call will re-load. Cached vectors are
        discarded too because their layout depends on the model.
        """
        with self._load_lock:
            self._st = None
        if self._cache is not None:
            self._cache.clear()

    # --- key + encode -------------------------------------------------------

    def _cache_key(self, text: str, *, kind: str) -> str:
        # `kind` separates query vs document encodings on asymmetric
        # models -- the same text encoded with `s2p_query` prompt vs.
        # no prompt yields DIFFERENT vectors and must not share a slot.
        # For symmetric models the query and document encoding are
        # bit-identical, so we collapse them onto kind="doc" so a
        # `embed_query(x)` shares a slot with `embed([x])[0]`.
        if kind == "query" and not self._query_kwargs:
            kind = "doc"
        return content_hash(
            self.model,
            self._dtype,
            kind,
            f"t{self._target_dim}" if self._target_dim is not None else "tNone",
            text,
        )

    def _truncate_and_renormalize(self, vec: list[float]) -> list[float]:
        """Truncate to `self._target_dim` and L2-renormalize when set."""
        if self._target_dim is None or self._target_dim == self._native_dim:
            return vec
        trimmed = vec[: self._target_dim]
        if not self._normalize:
            return trimmed
        norm = math.sqrt(sum(x * x for x in trimmed))
        if norm == 0.0:
            return trimmed
        return [x / norm for x in trimmed]

    def _encode(self, texts: list[str], *, extra: dict[str, Any] | None = None) -> list[list[float]]:
        self._ensure_loaded()
        kwargs: dict[str, Any] = {
            "batch_size": self._batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": self._normalize,
            "show_progress_bar": False,
        }
        if extra:
            kwargs.update(extra)
        vectors = self._st.encode(texts, **kwargs)
        out = [row.tolist() for row in vectors]
        if self._target_dim is not None and self._target_dim != self._native_dim:
            out = [self._truncate_and_renormalize(v) for v in out]
        return out

    # --- public surface -----------------------------------------------------

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
        # Every slot is filled by construction; flag any leftover None
        # rather than silently dropping the entry (would shorten the
        # returned list and break downstream zip-strict invariants).
        out: list[list[float]] = []
        for i, r in enumerate(results):
            if r is None:  # pragma: no cover - invariant violation
                raise RuntimeError(
                    f"LocalEmbedder.embed: cache fill left index {i} empty; "
                    f"this is a bug -- expected {len(text_list)} vectors."
                )
            out.append(r)
        return out

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query, applying asymmetric prompts when the
        model needs them.

        Symmetric models (bge-large, mxbai, ...) fall through to a
        regular encode -- the result is bit-identical to `embed([q])[0]`,
        so the hierarchy retriever can safely prefer this method when
        present without changing behavior on symmetric models. The
        cache slot is shared with the document side too, so warming
        one cache pre-warms the other (M-174).
        """
        cache_key = (
            self._cache_key(query, kind="query") if self._cache is not None else None
        )
        if self._cache is not None and cache_key is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        extra = dict(self._query_kwargs) if self._query_kwargs else None
        vec = self._encode([query], extra=extra)[0]
        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, vec)
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

    @property
    def is_loaded(self) -> bool:
        """True once the underlying SentenceTransformer has been built."""
        return self._st is not None

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
        tdim = (
            f"trunc={self._target_dim}"
            if self._target_dim is not None and self._target_dim != self._native_dim
            else "trunc=none"
        )
        return (
            f"local-embed/{self.model}/dim={self.dim}/{norm}/"
            f"device={self._device}/dtype={self._dtype}/{async_flag}/{tdim}/v3"
        )
