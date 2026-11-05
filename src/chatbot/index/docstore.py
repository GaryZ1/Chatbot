"""Chunk storage that is *not* searched: parent sections for small-to-big retrieval
(docs/02-chunking.md §2.12). Parents are fetched by id after children are matched.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from chatbot.core.interfaces import DocStore
from chatbot.core.types import Chunk


class InMemoryDocStore:
    """``DocStore`` backed by a dict, persisted as JSONL."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._by_doc: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()

    async def put(self, chunks: Sequence[Chunk]) -> None:
        async with self._lock:
            for c in chunks:
                if c.chunk_id not in self._chunks:
                    self._by_doc.setdefault(c.doc_id, []).append(c.chunk_id)
                self._chunks[c.chunk_id] = c

    async def replace_doc(self, doc_id: str, chunks: Sequence[Chunk]) -> None:
        await self.delete_doc(doc_id)
        await self.put(chunks)

    async def delete_doc(self, doc_id: str) -> int:
        async with self._lock:
            ids = self._by_doc.pop(doc_id, [])
            for i in ids:
                self._chunks.pop(i, None)
            return len(ids)

    async def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [self._chunks[i] for i in chunk_ids if i in self._chunks]

    async def by_doc(self, doc_id: str) -> list[Chunk]:
        chunks = [self._chunks[i] for i in self._by_doc.get(doc_id, []) if i in self._chunks]
        return sorted(chunks, key=lambda c: c.position)

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for c in self._chunks.values():
                fh.write(c.model_dump_json() + "\n")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> InMemoryDocStore:
        store = cls()
        path = Path(path)
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        c = Chunk.model_validate(json.loads(line))
                        store._chunks[c.chunk_id] = c
                        store._by_doc.setdefault(c.doc_id, []).append(c.chunk_id)
        return store


class CompositeDocStore:
    """Reads through several stores in order (e.g. parent store, then vector store)."""

    def __init__(self, stores: Sequence[DocStore]) -> None:
        self.stores = list(stores)

    async def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        found: dict[str, Chunk] = {}
        missing = list(chunk_ids)
        for s in self.stores:
            if not missing:
                break
            for c in await s.get(missing):
                found[c.chunk_id] = c
            missing = [i for i in missing if i not in found]
        return [found[i] for i in chunk_ids if i in found]

    async def by_doc(self, doc_id: str) -> list[Chunk]:
        seen: dict[str, Chunk] = {}
        for s in self.stores:
            for c in await s.by_doc(doc_id):
                seen.setdefault(c.chunk_id, c)
        return sorted(seen.values(), key=lambda c: (c.position, c.kind.value))
