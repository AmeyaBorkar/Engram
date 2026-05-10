"""Local sentence-transformers embedder.

Behind the `[bench]` extra (or any environment with `sentence-transformers`
installed). The point is: real-quality embeddings without an API key,
running on the user's GPU or CPU.

Defaults to `BAAI/bge-large-en-v1.5` -- 330 M params, 1024-dim, top-tier
on MTEB retrieval (~54). Override with any sentence-transformers model
id. Some other strong choices:

  * `BAAI/bge-base-en-v1.5`            -- 110M, 768-dim, faster
  * `BAAI/bge-m3`                      -- 568M, multilingual, 1024-dim
  * `mixedbread-ai/mxbai-embed-large-v1` -- ~330M, 1024-dim, MTEB ~54.4
  * `intfloat/e5-large-v2`             -- ~335M, 1024-dim
  * `nomic-ai/nomic-embed-text-v1.5`   -- 137M, 768-dim, matryoshka

GPU auto-detect: `torch.cuda.is_available()` chooses CUDA; CPU falls
out otherwise. Override explicitly via `device="cuda:1"` etc. Embeddings
are L2-normalized at encode time so cosine similarity reduces to a
dot product in storage.

First call to `embed` triggers model download into the HuggingFace
cache (default `~/.cache/huggingface/hub/`). For `BAAI/bge-large-en-v1.5`
that's roughly 1.3 GiB on disk; the download happens once.
"""

from __future__ import annotations

from collections.abc import Sequence

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is not installed. Install with: "
        "pip install 'engram[bench]'   # or: pip install sentence-transformers"
    ) from _exc


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
        batch_size: int = 32,
        dim: int | None = None,
        normalize: bool = True,
    ) -> None:
        chosen_device = device if device is not None else _detect_device()
        self._st = _SentenceTransformer(model, device=chosen_device)
        self.model = model
        self._device = chosen_device
        self._batch_size = batch_size
        self._normalize = normalize
        native_dim = int(self._st.get_sentence_embedding_dimension() or 0)
        if dim is not None and dim != native_dim:
            raise ValueError(f"requested dim={dim} does not match model native dim={native_dim}")
        self.dim = native_dim

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
        # Device flows into the hash so a CPU run and a CUDA run of the
        # same model land in distinct manifest rows -- helpful when the
        # determinism floor between the two differs (mixed precision,
        # cuBLAS reductions, ...).
        return f"local-embed/{self.model}/dim={self.dim}/{norm}/device={self._device}/v1"
