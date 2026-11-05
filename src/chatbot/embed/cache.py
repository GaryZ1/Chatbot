"""Query-embedding cache (docs/04-embeddings.md §8: "cache query embeddings keyed by
normalized text").

Query embedding latency adds directly to p95, and popular questions repeat, so
:class:`CachingEmbedder` wraps any embedder with an in-process LRU keyed by
``(model key, normalised query)``. The model key includes name, version and dims,
so a model upgrade can never serve stale vectors. Document embedding is passed
through uncached (ingestion handles its own dedup via content hashes).
"""

from __future__ import annotations

import asyncio
import threading
import unicodedata
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from chatbot.core.interfaces import Embedder


def normalize_query(text: str, *, casefold: bool = False) -> str:
    """NFKC + whitespace collapse (+ optional casefold) — the cache key text."""
    norm = " ".join(unicodedata.normalize("NFKC", text).split())
    return norm.casefold() if casefold else norm


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class QueryEmbeddingCache:
    """Thread-safe LRU of ``key -> vector``. Stored vectors are read-only copies."""

    def __init__(self, maxsize: int = 10_000) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._data: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get(self, model_key: str, text: str) -> np.ndarray | None:
        key = (model_key, text)
        with self._lock:
            vec = self._data.get(key)
            if vec is None:
                self.stats.misses += 1
                return None
            self._data.move_to_end(key)
            self.stats.hits += 1
            return vec

    def put(self, model_key: str, text: str, vector: np.ndarray) -> None:
        vec = np.array(vector, dtype=np.float32, copy=True)
        vec.setflags(write=False)
        with self._lock:
            self._data[(model_key, text)] = vec
            self._data.move_to_end((model_key, text))
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def _model_key(embedder: Embedder) -> str:
    info = getattr(embedder, "info", None)
    key = getattr(info, "key", None)
    if isinstance(key, str):
        return key
    return f"{embedder.name}:{embedder.dim}"


class CachingEmbedder:
    """Embedder decorator caching :meth:`embed_query`; concurrent misses are coalesced."""

    def __init__(
        self,
        inner: Embedder,
        cache: QueryEmbeddingCache | None = None,
        *,
        casefold: bool = False,
    ) -> None:
        self.inner = inner
        self.cache = cache or QueryEmbeddingCache()
        self.casefold = casefold
        self.name = inner.name
        self._model_key = _model_key(inner)
        self._inflight: dict[str, asyncio.Future[np.ndarray]] = {}

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def info(self) -> object:
        return getattr(self.inner, "info", None)

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return await self.inner.embed_documents(texts)

    async def embed_query(self, text: str) -> np.ndarray:
        norm = normalize_query(text, casefold=self.casefold)
        cached = self.cache.get(self._model_key, norm)
        if cached is not None:
            return cached
        pending = self._inflight.get(norm)
        if pending is not None:
            return await asyncio.shield(pending)
        fut: asyncio.Future[np.ndarray] = asyncio.get_running_loop().create_future()
        self._inflight[norm] = fut
        try:
            vec = await self.inner.embed_query(norm)
            self.cache.put(self._model_key, norm, vec)
            result = self.cache.get(self._model_key, norm)
            out = result if result is not None else vec
            fut.set_result(out)
            return out
        except asyncio.CancelledError:
            fut.cancel()
            raise
        except Exception as exc:
            fut.set_exception(exc)
            fut.exception()  # mark retrieved so no "never retrieved" warning
            raise
        finally:
            self._inflight.pop(norm, None)
