"""Chroma baseline.

Wraps an in-memory `chromadb` collection in the `Retriever` protocol so
the benchmark harness can drive it the same way it drives Engram. Pass
an `EmbeddingProvider` to make the comparison apples-to-apples (both
sides use the same embeddings); without one, Chroma's default ONNX
embedder is used.

Behind the `[bench]` extra. Tests skip when `chromadb` isn't installed.
"""

from __future__ import annotations

import uuid
from typing import Any

from engram.bench import Hit
from engram.providers import EmbeddingProvider


class ChromaRetriever:
    """`engram.bench.Retriever` for an in-memory Chroma collection."""

    name: str = "chroma"

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        *,
        collection_name: str | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "chromadb is not installed. Install with: pip install 'engram[bench]'"
            ) from exc

        # Each retriever instance gets its own ephemeral client + uniquely-
        # named collection so multiple instances within one process (e.g. one
        # test per Retriever) don't collide.
        # M-166: keep a handle on the client so callers can `close()`
        # it later. EphemeralClient holds a sqlite file open in the
        # background; tests / suites that build many retrievers and
        # never let them be garbage-collected leak file handles
        # (visible as "too many open files" on long CI runs).
        self._client = chromadb.EphemeralClient()
        name = collection_name or f"engram-bench-{uuid.uuid4().hex[:8]}"
        if embedder is None:
            self._collection: Any = self._client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._collection = self._client.create_collection(
                name=name,
                embedding_function=_EmbeddingFunctionAdapter(embedder),
                metadata={"hnsw:space": "cosine"},
            )

    def add(self, content: str, doc_id: str | None = None) -> str:
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        self._collection.add(documents=[content], ids=[doc_id])
        return doc_id

    def query(self, query: str, k: int) -> list[Hit]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        result = self._collection.query(query_texts=[query], n_results=k)
        ids = result["ids"][0] if result.get("ids") else []
        docs = result["documents"][0] if result.get("documents") else []
        distances = result["distances"][0] if result.get("distances") else []
        out: list[Hit] = []
        for i, doc_id in enumerate(ids):
            content = docs[i] if i < len(docs) else ""
            distance = distances[i] if i < len(distances) else 1.0
            # Cosine distance in chroma is `1 - cosine_similarity` (so similarity
            # is `1 - distance`).
            out.append(Hit(id=str(doc_id), content=str(content), score=1.0 - float(distance)))
        return out

    def close(self) -> None:
        """Release the underlying EphemeralClient + collection handle.

        M-166: chromadb's ``EphemeralClient`` holds an in-process
        sqlite handle plus an HNSW segment in memory; tests / suites
        that built many ``ChromaRetriever`` instances without ever
        releasing them leaked both. Calling ``close()`` deletes the
        collection (best-effort -- some chromadb versions don't
        expose the API) and drops our references so the GC can
        reclaim the handles immediately.
        """
        try:
            self._client.delete_collection(self._collection.name)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
        # Drop the references so any next-line `del retriever` collapses
        # to a pure refcount drop with no live cycle.
        self._collection = None  # type: ignore[assignment]
        self._client = None  # type: ignore[assignment]


class _EmbeddingFunctionAdapter:
    """Adapt an `EmbeddingProvider` to chromadb's expected interface.

    Chromadb 0.5+ expects custom embedding functions to expose:
      - `__call__(input)`         — embed documents at index time
      - `embed_query(input)`      — embed queries at query time
      - `name()`                  — stable identifier for the function
      - `is_legacy()`             — back-compat flag (`False` for current API)
    """

    is_legacy_attr: bool = False

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self._embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed(input)

    def name(self) -> str:
        return f"engram-{self._embedder.name}-{self._embedder.model}"

    def is_legacy(self) -> bool:
        return False
