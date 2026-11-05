"""Blue/green re-indexing behind aliases (docs/05 §3, docs/04-embeddings.md §8).

Migration recipe: create a new versioned index → dual-write / backfill →
shadow-evaluate → **switch the alias atomically** → drop the old index. Readers
hold an :class:`AliasedVectorStore`, which resolves the alias on every call, so
a switch is zero-downtime and needs no restart.

:class:`AliasManager` manages in-process stores (e.g. :class:`InMemoryVectorStore`)
created by a factory and is also the abstraction used for external engines: pass
``on_switch`` (e.g. :meth:`QdrantVectorStore.switch_alias`) to mirror alias moves
into the engine.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np

from chatbot.core.filters import Filter
from chatbot.core.interfaces import Embedder, VectorStore
from chatbot.core.types import Chunk, Principal, ScoredChunk
from chatbot.index.schema import IndexSchema

log = logging.getLogger(__name__)

StoreFactory = Callable[[IndexSchema], VectorStore]
SwitchHook = Callable[[str, str], Awaitable[None]]


@dataclass
class AliasSwitch:
    alias: str
    old: str | None
    new: str


class AliasManager:
    def __init__(self, factory: StoreFactory, *, on_switch: SwitchHook | None = None) -> None:
        self.factory = factory
        self.on_switch = on_switch
        self._stores: dict[str, VectorStore] = {}
        self._schemas: dict[str, IndexSchema] = {}
        self._aliases: dict[str, str] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ indexes
    def create_index(self, schema: IndexSchema) -> VectorStore:
        name = schema.collection_name
        if name in self._stores:
            raise ValueError(f"index {name!r} already exists")
        store = self.factory(schema)
        self._stores[name] = store
        self._schemas[name] = schema
        log.info("created index %s", name)
        return store

    def store(self, name: str) -> VectorStore:
        try:
            return self._stores[name]
        except KeyError:
            raise KeyError(f"unknown index {name!r}") from None

    def schema(self, name: str) -> IndexSchema:
        return self._schemas[name]

    def indexes(self) -> list[str]:
        return list(self._stores)

    async def backfill(
        self,
        name: str,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        *,
        batch_size: int = 128,
    ) -> int:
        """Embed and upsert ``chunks`` into index ``name`` (the green side)."""
        store = self.store(name)
        n = 0
        for i in range(0, len(chunks), batch_size):
            batch = list(chunks[i : i + batch_size])
            vecs = await embedder.embed_documents([c.index_text for c in batch])
            await store.upsert(batch, vecs)
            n += len(batch)
        log.info("backfilled %d chunks into %s", n, name)
        return n

    # ------------------------------------------------------------------ aliases
    def resolve(self, alias: str) -> str:
        try:
            return self._aliases[alias]
        except KeyError:
            raise KeyError(f"alias {alias!r} is not bound") from None

    def resolve_store(self, alias: str) -> VectorStore:
        return self._stores[self.resolve(alias)]

    async def switch(self, alias: str, name: str) -> AliasSwitch:
        """Point ``alias`` at index ``name`` in one assignment (atomic for readers)."""
        if name not in self._stores:
            raise KeyError(f"unknown index {name!r}")
        async with self._lock:
            old = self._aliases.get(alias)
            if self.on_switch is not None:
                await self.on_switch(alias, name)
            self._aliases[alias] = name
        log.info("alias %s: %s -> %s", alias, old, name)
        return AliasSwitch(alias=alias, old=old, new=name)

    async def drop(self, name: str) -> None:
        """Drop an index; refuses while any alias still points at it."""
        async with self._lock:
            bound = [a for a, n in self._aliases.items() if n == name]
            if bound:
                raise ValueError(f"index {name!r} is still aliased by {bound}")
            self._stores.pop(name, None)
            self._schemas.pop(name, None)

    async def migrate(
        self,
        current: IndexSchema,
        new_schema: IndexSchema,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        *,
        validate: Callable[[VectorStore], Awaitable[bool]] | None = None,
        drop_old: bool = True,
    ) -> AliasSwitch:
        """Full blue/green: create → backfill → validate (shadow eval) → switch → drop old."""
        self.create_index(new_schema)
        await self.backfill(new_schema.collection_name, chunks, embedder)
        if validate is not None and not await validate(self.store(new_schema.collection_name)):
            raise RuntimeError(f"validation failed for {new_schema.collection_name}; alias unchanged")
        sw = await self.switch(current.alias_name, new_schema.collection_name)
        if drop_old and sw.old and sw.old != sw.new:
            await self.drop(sw.old)
        return sw

    def view(self, alias: str) -> AliasedVectorStore:
        return AliasedVectorStore(self, alias)


class AliasedVectorStore:
    """A :class:`VectorStore` proxy that follows an alias on every call."""

    def __init__(self, manager: AliasManager, alias: str) -> None:
        self.manager = manager
        self.alias = alias

    @property
    def target(self) -> VectorStore:
        return self.manager.resolve_store(self.alias)

    async def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        await self.target.upsert(chunks, vectors)

    async def delete_doc(self, doc_id: str) -> int:
        return await self.target.delete_doc(doc_id)

    async def search(
        self,
        vector: np.ndarray,
        k: int,
        principal: Principal | None,
        flt: Filter | None = None,
    ) -> list[ScoredChunk]:
        return await self.target.search(vector, k, principal, flt)

    async def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return await self.target.get(chunk_ids)

    async def by_doc(self, doc_id: str) -> list[Chunk]:
        return await self.target.by_doc(doc_id)

    async def count(self) -> int:
        return await self.target.count()
