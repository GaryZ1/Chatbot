"""Bi-encoder base class (docs/04-embeddings.md §1, §8).

Every concrete embedder inherits :class:`BaseEmbedder`, which owns the parts that
are easy to get silently wrong:

* **Asymmetric prefixes** (§1). Queries and documents get *different* instructions
  (``"query: "`` / ``"passage: "`` for E5, ``"search_query: "`` for Nomic, ...).
  Forgetting the query instruction is the classic silent bug (docs/07-retrieval.md
  §6), so the prefixes are explicit constructor arguments. ``query_prefix=None``
  means "use the known default for this model" (see :data:`KNOWN_PREFIXES`); pass
  ``""`` to state explicitly that the model needs none.
* **Batching** (§8): documents are embedded in batches of ``batch_size`` (32–256).
* **L2 normalisation** (§8): vectors are unit-normalised so cosine == dot product,
  matching the vector store metric.
* **Vector versioning** (§8): :class:`EmbeddingInfo` (name, version, dims) is stored
  with every vector; never mix models in one index.
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

#: Known (query_prefix, document_prefix) defaults by model-name substring (lower-case).
#: First match wins, so more specific patterns come first.
KNOWN_PREFIXES: list[tuple[str, tuple[str, str]]] = [
    ("multilingual-e5", ("query: ", "passage: ")),
    ("e5-mistral", ("Instruct: Given a question, retrieve passages that answer it\nQuery: ", "")),
    ("e5-", ("query: ", "passage: ")),
    ("nomic-embed", ("search_query: ", "search_document: ")),
    ("bge-m3", ("", "")),
    ("bge-", ("Represent this sentence for searching relevant passages: ", "")),
    ("mxbai-embed", ("Represent this sentence for searching relevant passages: ", "")),
    ("snowflake-arctic-embed", ("Represent this sentence for searching relevant passages: ", "")),
    (
        "qwen3-embedding",
        ("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", ""),
    ),
    ("gte-qwen", ("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", "")),
]


def default_prefixes(model_name: str) -> tuple[str, str] | None:
    """Return the known ``(query_prefix, document_prefix)`` for ``model_name`` or ``None``."""
    low = model_name.lower()
    for pattern, prefixes in KNOWN_PREFIXES:
        if pattern in low:
            return prefixes
    return None


@dataclass(frozen=True)
class EmbeddingInfo:
    """Identity of a vector space. Stored on every vector / collection (§8)."""

    name: str
    version: str
    dims: int
    metric: str = "cosine"
    normalized: bool = True

    @property
    def key(self) -> str:
        """Compact identifier, e.g. ``hashing@1:384``; part of collection names and cache keys."""
        return f"{self.name}@{self.version}:{self.dims}"

    def compatible_with(self, other: EmbeddingInfo) -> bool:
        return (self.name, self.version, self.dims, self.metric) == (
            other.name,
            other.version,
            other.dims,
            other.metric,
        )


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation (works for 1-D and 2-D arrays); zero rows stay zero."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        return arr / n if n > eps else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > eps, norms, 1.0)
    return (arr / norms).astype(np.float32)


class BaseEmbedder(abc.ABC):
    """Template for embedders implementing :class:`chatbot.core.interfaces.Embedder`.

    Subclasses implement :meth:`_embed_batch`, which receives already-prefixed texts.
    """

    def __init__(
        self,
        *,
        name: str,
        dim: int,
        version: str = "1",
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        max_chars: int | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        known = default_prefixes(name)
        if query_prefix is None:
            query_prefix = known[0] if known else ""
        if document_prefix is None:
            document_prefix = known[1] if known else ""
        if known and known[0] and not query_prefix:
            log.warning(
                "embedder %s normally needs query prefix %r but query_prefix='' was set; "
                "retrieval quality may silently degrade",
                name,
                known[0],
            )
        self.name = name
        self._dim = dim
        self.version = version
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_chars = max_chars

    # -- identity ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return self._dim

    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            name=self.name, version=self.version, dims=self.dim, normalized=self.normalize
        )

    # -- implementation hook -------------------------------------------------------
    @abc.abstractmethod
    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        """Embed one batch of prefixed texts → array of shape ``(len(texts), dim)``."""

    # -- public API ----------------------------------------------------------------
    def _prepare(self, text: str, prefix: str) -> str:
        if self.max_chars is not None and len(text) > self.max_chars:
            text = text[: self.max_chars]
        return prefix + text

    def _finish(self, arr: np.ndarray, expected_rows: int) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float32)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        if out.shape != (expected_rows, self.dim):
            raise ValueError(
                f"{self.name} returned shape {out.shape}, expected ({expected_rows}, {self.dim})"
            )
        return l2_normalize(out) if self.normalize else out

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents in batches with the document prefix; returns ``(n, dim)`` float32."""
        if isinstance(texts, str):
            raise TypeError("embed_documents expects a sequence of strings, not a string")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [self._prepare(t, self.document_prefix) for t in texts[start : start + self.batch_size]]
            parts.append(self._finish(await self._embed_batch(batch, is_query=False), len(batch)))
        return np.vstack(parts)

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a query with the *query* prefix; returns a ``(dim,)`` float32 vector."""
        if not isinstance(text, str):
            raise TypeError("embed_query expects a string")
        arr = await self._embed_batch([self._prepare(text, self.query_prefix)], is_query=True)
        return self._finish(arr, 1)[0]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dim}, version={self.version!r})"
