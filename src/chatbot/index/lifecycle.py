"""Index lifecycle: idempotent document (re)indexing (docs/05-indexing-vector-stores.md §3–4).

:class:`IndexWriter.index_document` implements the update protocol:

1. **skip** if the document's ``content_hash`` (and version) is unchanged — no
   re-embedding cost;
2. **delete by ``doc_id``** in the vector store and the sparse index (removes
   stale chunks whose positions no longer exist);
3. embed the chunks' ``index_text`` (contextual text) in batches and **upsert** by
   ``chunk_id`` into both indexes, stamping ``is_latest``, ``content_hash``,
   ``embedding_model`` and ``indexed_at`` metadata; with ``keep_history`` older
   versions are retained with ``is_latest = False`` for "as of" queries;
4. record **freshness lag** (source update → searchable) as an SLO metric.

Tombstoned documents (``Document.deleted``) are removed. :meth:`compact` runs the
stores' compaction (HNSW tombstones, §4) plus an optional hook.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

import numpy as np

from chatbot.core.interfaces import Embedder, SparseIndex, VectorStore
from chatbot.core.types import Chunk, Document, utcnow

log = logging.getLogger(__name__)


class IndexAction(StrEnum):
    INDEXED = "indexed"
    SKIPPED = "skipped"
    DELETED = "deleted"


@dataclass
class IndexResult:
    doc_id: str
    action: IndexAction
    chunks_written: int = 0
    chunks_deleted: int = 0
    freshness_lag_s: float | None = None
    duration_s: float = 0.0


class FreshnessTracker:
    """Rolling window of source-update → searchable lags (seconds)."""

    def __init__(self, window: int = 1000) -> None:
        self._lags: deque[float] = deque(maxlen=window)

    def record(self, updated_at: datetime | None, indexed_at: datetime | None = None) -> float | None:
        if updated_at is None:
            return None
        indexed_at = indexed_at or utcnow()
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=indexed_at.tzinfo)
        lag = max((indexed_at - updated_at).total_seconds(), 0.0)
        self._lags.append(lag)
        return lag

    def percentile(self, p: float) -> float | None:
        if not self._lags:
            return None
        data = sorted(self._lags)
        idx = min(len(data) - 1, max(0, round(p / 100 * (len(data) - 1))))
        return data[idx]

    def summary(self) -> dict[str, float]:
        if not self._lags:
            return {}
        return {
            "count": float(len(self._lags)),
            "mean_s": statistics.fmean(self._lags),
            "p50_s": self.percentile(50) or 0.0,
            "p95_s": self.percentile(95) or 0.0,
            "max_s": max(self._lags),
        }


@dataclass
class _DocState:
    version: str
    content_hash: str


CompactionHook = Callable[[], Awaitable[None]]


@dataclass
class IndexWriter:
    embedder: Embedder
    vector_store: VectorStore
    sparse_index: SparseIndex | None = None
    batch_size: int = 64
    keep_history: bool = False
    compaction_hook: CompactionHook | None = None
    freshness: FreshnessTracker = field(default_factory=FreshnessTracker)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._state: dict[str, _DocState] = {}
        self._doc_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def _embedding_key(self) -> str:
        info = getattr(self.embedder, "info", None)
        key = getattr(info, "key", None)
        return key if isinstance(key, str) else f"{self.embedder.name}:{self.embedder.dim}"

    async def _known_state(self, doc_id: str) -> tuple[_DocState | None, list[Chunk]]:
        existing = await self.vector_store.by_doc(doc_id)
        state = self._state.get(doc_id)
        if state is None:
            for c in existing:
                if c.metadata.get("is_latest", True) and "content_hash" in c.metadata:
                    state = _DocState(c.doc_version, str(c.metadata["content_hash"]))
                    break
        return state, existing

    def _stamp(self, chunk: Chunk, doc: Document, *, is_latest: bool, now: datetime) -> Chunk:
        meta: dict[str, Any] = dict(chunk.metadata)
        meta["is_latest"] = is_latest
        meta["content_hash"] = doc.content_hash
        meta["embedding_model"] = self._embedding_key
        meta["indexed_at"] = now.isoformat()
        if doc.updated_at is not None:
            meta.setdefault("updated_at", doc.updated_at.isoformat())
        meta.setdefault("doc_type", doc.doc_type)
        meta.setdefault("language", doc.language)
        return chunk.model_copy(update={"metadata": meta})

    async def _embed(self, chunks: Sequence[Chunk]) -> np.ndarray:
        parts = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            parts.append(await self.embedder.embed_documents([c.index_text for c in batch]))
        return np.vstack(parts) if parts else np.zeros((0, self.embedder.dim), dtype=np.float32)

    def _lock(self, doc_id: str) -> asyncio.Lock:
        return self._doc_locks.setdefault(doc_id, asyncio.Lock())

    # ------------------------------------------------------------------ API
    async def index_document(self, doc: Document, chunks: Sequence[Chunk], *, force: bool = False) -> IndexResult:
        """Idempotently (re)index ``doc`` with its ``chunks``."""
        if doc.deleted:
            return await self.delete_document(doc.doc_id)
        bad = [c.chunk_id for c in chunks if c.doc_id != doc.doc_id]
        if bad:
            raise ValueError(f"chunks {bad[:3]} do not belong to document {doc.doc_id!r}")
        t0 = time.perf_counter()
        async with self._lock(doc.doc_id):
            state, existing = await self._known_state(doc.doc_id)
            if (
                not force
                and state is not None
                and state.content_hash == doc.content_hash
                and state.version == doc.version
            ):
                log.debug("skip unchanged document %s", doc.doc_id)
                return IndexResult(doc.doc_id, IndexAction.SKIPPED, duration_s=time.perf_counter() - t0)

            now = utcnow()
            new_chunks = [self._stamp(c, doc, is_latest=True, now=now) for c in chunks]
            vectors = await self._embed(new_chunks)  # embed before deleting: no empty window on failure

            history: list[Chunk] = []
            history_vecs: np.ndarray | None = None
            if self.keep_history:
                new_ids = {c.chunk_id for c in new_chunks}
                old = [c for c in existing if c.doc_version != doc.version and c.chunk_id not in new_ids]
                history = [
                    c.model_copy(update={"metadata": {**c.metadata, "is_latest": False}}) for c in old
                ]
                if history:
                    getter = getattr(self.vector_store, "get_vectors", None)
                    stored: dict[str, np.ndarray] = await getter([c.chunk_id for c in history]) if getter else {}
                    if len(stored) == len(history):
                        history_vecs = np.vstack([stored[c.chunk_id] for c in history])
                    else:
                        history_vecs = await self._embed(history)

            deleted = await self.vector_store.delete_doc(doc.doc_id)
            if self.sparse_index is not None:
                await self.sparse_index.delete_doc(doc.doc_id)
            all_chunks = [*history, *new_chunks]
            all_vecs = vectors if history_vecs is None else np.vstack([history_vecs, vectors])
            if all_chunks:
                await self.vector_store.upsert(all_chunks, all_vecs)
                if self.sparse_index is not None:
                    await self.sparse_index.add(all_chunks)
            self._state[doc.doc_id] = _DocState(doc.version, doc.content_hash)
            lag = self.freshness.record(doc.updated_at, now)
        return IndexResult(
            doc.doc_id,
            IndexAction.INDEXED,
            chunks_written=len(new_chunks),
            chunks_deleted=max(deleted - len(history), 0),
            freshness_lag_s=lag,
            duration_s=time.perf_counter() - t0,
        )

    async def delete_document(self, doc_id: str) -> IndexResult:
        t0 = time.perf_counter()
        async with self._lock(doc_id):
            n = await self.vector_store.delete_doc(doc_id)
            if self.sparse_index is not None:
                await self.sparse_index.delete_doc(doc_id)
            self._state.pop(doc_id, None)
        return IndexResult(doc_id, IndexAction.DELETED, chunks_deleted=n, duration_s=time.perf_counter() - t0)

    async def index_many(
        self, items: Sequence[tuple[Document, Sequence[Chunk]]], *, concurrency: int = 4
    ) -> list[IndexResult]:
        sem = asyncio.Semaphore(max(1, concurrency))

        async def one(doc: Document, chunks: Sequence[Chunk]) -> IndexResult:
            async with sem:
                return await self.index_document(doc, chunks)

        return list(await asyncio.gather(*(one(d, c) for d, c in items)))

    async def compact(self) -> None:
        """Run store compaction (tombstones) and the user-supplied hook, if any."""
        for store in (self.vector_store, self.sparse_index):
            fn = getattr(store, "compact", None)
            if fn is not None:
                await fn()
        if self.compaction_hook is not None:
            await self.compaction_hook()
