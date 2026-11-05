"""Exact (flat) in-memory vector store + doc store (docs/05-indexing-vector-stores.md §1).

Flat search is 100% recall and the ground truth for ANN recall tests; it is the
right choice below ~100k vectors and for tests. Security-relevant behaviour
matches production stores:

* the ACL / tenant / metadata **pre-filter runs before scoring and top-K**
  (:func:`chatbot.core.filters.visible`), so a restricted chunk can never occupy a
  slot and ``k`` results are still returned from the visible set (docs/07 §6);
* upserts are keyed by ``chunk_id`` (§4) and ``delete_doc`` removes every version;
* persistence: ``vectors.npz`` + ``chunks.jsonl`` + ``meta.json`` in a directory.

Concurrency: writers are serialised with an :class:`asyncio.Lock`; internal state
is guarded by a :class:`threading.RLock` so ``save`` from a worker thread sees a
consistent snapshot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from chatbot.core.filters import Filter, visible
from chatbot.core.types import Chunk, Principal, ScoredChunk
from chatbot.embed.base import EmbeddingInfo, l2_normalize

log = logging.getLogger(__name__)

_FORMAT_VERSION = 1


class InMemoryVectorStore:
    """Implements :class:`~chatbot.core.interfaces.VectorStore` and ``DocStore``."""

    def __init__(
        self,
        dim: int,
        *,
        metric: str = "cosine",
        embedding: EmbeddingInfo | None = None,
        name: str = "memory",
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if metric not in {"cosine", "dot"}:
            raise ValueError("metric must be 'cosine' or 'dot'")
        if embedding is not None and embedding.dims != dim:
            raise ValueError("embedding.dims does not match dim")
        self.dim = dim
        self.metric = metric
        self.embedding = embedding
        self.name = name
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._by_doc: dict[str, set[str]] = {}
        self._ids: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._dirty = False
        self._state = threading.RLock()
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------ internals
    def _prep(self, vectors: np.ndarray) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(f"vector dim {arr.shape[1]} != store dim {self.dim}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("vectors contain NaN or inf")
        return l2_normalize(arr) if self.metric == "cosine" else arr

    def _snapshot(self) -> tuple[list[str], np.ndarray]:
        with self._state:
            if self._dirty:
                self._ids = list(self._vectors)
                self._matrix = (
                    np.vstack([self._vectors[i] for i in self._ids])
                    if self._ids
                    else np.zeros((0, self.dim), dtype=np.float32)
                )
                self._dirty = False
            return self._ids, self._matrix

    def _put(self, chunk: Chunk, vec: np.ndarray) -> None:
        old = self._chunks.get(chunk.chunk_id)
        if old is not None and old.doc_id != chunk.doc_id:
            self._by_doc.get(old.doc_id, set()).discard(chunk.chunk_id)
        self._chunks[chunk.chunk_id] = chunk
        self._vectors[chunk.chunk_id] = vec
        self._by_doc.setdefault(chunk.doc_id, set()).add(chunk.chunk_id)
        self._dirty = True

    # ------------------------------------------------------------------ writes
    async def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        arr = self._prep(vectors)
        if arr.shape[0] != len(chunks):
            raise ValueError(f"{len(chunks)} chunks but {arr.shape[0]} vectors")
        async with self._write_lock:
            with self._state:
                for chunk, vec in zip(chunks, arr, strict=True):
                    self._put(chunk, vec.copy())

    async def delete_doc(self, doc_id: str) -> int:
        async with self._write_lock:
            with self._state:
                ids = self._by_doc.pop(doc_id, set())
                for cid in ids:
                    self._chunks.pop(cid, None)
                    self._vectors.pop(cid, None)
                if ids:
                    self._dirty = True
                return len(ids)

    async def delete(self, chunk_ids: Sequence[str]) -> int:
        async with self._write_lock:
            with self._state:
                n = 0
                for cid in chunk_ids:
                    chunk = self._chunks.pop(cid, None)
                    if chunk is None:
                        continue
                    self._vectors.pop(cid, None)
                    self._by_doc.get(chunk.doc_id, set()).discard(cid)
                    n += 1
                if n:
                    self._dirty = True
                return n

    async def compact(self) -> None:
        """Rebuild the scan matrix and drop empty doc entries (tombstone compaction, §4)."""
        with self._state:
            self._by_doc = {d: ids for d, ids in self._by_doc.items() if ids}
            self._dirty = True
        self._snapshot()

    # ------------------------------------------------------------------ reads
    async def search(
        self,
        vector: np.ndarray,
        k: int,
        principal: Principal | None,
        flt: Filter | None = None,
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []
        q = self._prep(vector)[0]
        ids, matrix = self._snapshot()
        with self._state:
            chunks = self._chunks
            rows = [i for i, cid in enumerate(ids) if cid in chunks and visible(chunks[cid], principal, flt)]
            if not rows:
                return []
            row_idx = np.asarray(rows, dtype=np.int64)
            scores = matrix[row_idx] @ q
            kk = min(k, scores.shape[0])
            top = np.argpartition(-scores, kk - 1)[:kk]
            top = top[np.argsort(-scores[top], kind="stable")]
            return [
                ScoredChunk(
                    chunk=chunks[ids[row_idx[t]]],
                    score=float(scores[t]),
                    source=self.name,
                    stage_scores={self.name: float(scores[t])},
                )
                for t in top
            ]

    async def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        with self._state:
            return [self._chunks[c] for c in chunk_ids if c in self._chunks]

    async def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, np.ndarray]:
        with self._state:
            return {c: self._vectors[c].copy() for c in chunk_ids if c in self._vectors}

    async def by_doc(self, doc_id: str) -> list[Chunk]:
        with self._state:
            chunks = [self._chunks[c] for c in self._by_doc.get(doc_id, ())]
        return sorted(chunks, key=lambda c: (c.kind.value, c.position, c.chunk_id))

    async def neighbours(
        self,
        chunk_id: str,
        window: int = 1,
        principal: Principal | None = None,
        flt: Filter | None = None,
    ) -> list[Chunk]:
        """Same-document, same-kind chunks at positions ``i±window`` (excluding ``i``)."""
        with self._state:
            hit = self._chunks.get(chunk_id)
            if hit is None:
                return []
            siblings = [self._chunks[c] for c in self._by_doc.get(hit.doc_id, ())]
        out = [
            c
            for c in siblings
            if c.kind == hit.kind
            and c.doc_version == hit.doc_version
            and c.chunk_id != chunk_id
            and abs(c.position - hit.position) <= window
            and visible(c, principal, flt)
        ]
        return sorted(out, key=lambda c: c.position)

    async def count(self) -> int:
        with self._state:
            return len(self._chunks)

    async def all_chunks(self) -> list[Chunk]:
        with self._state:
            return list(self._chunks.values())

    def doc_ids(self) -> list[str]:
        with self._state:
            return [d for d, ids in self._by_doc.items() if ids]

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        """Write ``meta.json``, ``vectors.npz`` and ``chunks.jsonl`` atomically-ish."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with self._state:
            ids = list(self._chunks)
            matrix = (
                np.vstack([self._vectors[i] for i in ids])
                if ids
                else np.zeros((0, self.dim), dtype=np.float32)
            )
            lines = [self._chunks[i].model_dump_json() for i in ids]
        meta = {
            "format": _FORMAT_VERSION,
            "dim": self.dim,
            "metric": self.metric,
            "name": self.name,
            "count": len(ids),
            "embedding": None
            if self.embedding is None
            else {
                "name": self.embedding.name,
                "version": self.embedding.version,
                "dims": self.embedding.dims,
                "metric": self.embedding.metric,
                "normalized": self.embedding.normalized,
            },
        }
        tmp_vec = path / "vectors.tmp.npz"
        np.savez(tmp_vec, ids=np.asarray(ids, dtype=np.str_), vectors=matrix)
        tmp_vec.replace(path / "vectors.npz")
        tmp_chunks = path / "chunks.jsonl.tmp"
        tmp_chunks.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        tmp_chunks.replace(path / "chunks.jsonl")
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("saved %d vectors to %s", len(ids), path)

    @classmethod
    def load(cls, directory: str | Path) -> InMemoryVectorStore:
        path = Path(directory)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format") != _FORMAT_VERSION:
            raise ValueError(f"unsupported store format {meta.get('format')!r}")
        emb = meta.get("embedding")
        store = cls(
            int(meta["dim"]),
            metric=meta["metric"],
            name=meta.get("name", "memory"),
            embedding=EmbeddingInfo(**emb) if emb else None,
        )
        with np.load(path / "vectors.npz", allow_pickle=False) as data:
            ids = [str(i) for i in data["ids"].tolist()]
            matrix = np.asarray(data["vectors"], dtype=np.float32)
        chunks: dict[str, Chunk] = {}
        with (path / "chunks.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    c = Chunk.model_validate_json(line)
                    chunks[c.chunk_id] = c
        if len(ids) != matrix.shape[0] or set(ids) != set(chunks):
            raise ValueError("corrupt store: ids, vectors and chunks disagree")
        with store._state:
            for cid, vec in zip(ids, matrix, strict=True):
                store._put(chunks[cid], vec.copy())
        return store
