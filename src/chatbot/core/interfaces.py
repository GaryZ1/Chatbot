"""Stage interfaces. Every component sits behind one of these so it can be swapped
and A/B-tested through the evaluation harness (docs/16-build-plan.md, "Target stack").
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from chatbot.core.filters import Filter
from chatbot.core.types import Chunk, Document, Principal, ScoredChunk


@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    """Bi-encoder. Queries and documents are embedded asymmetrically (query prefix!)."""

    name: str
    dim: int

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    async def embed_query(self, text: str) -> np.ndarray: ...


@runtime_checkable
class DocStore(Protocol):
    """Lookup of chunks by id / document, used for parent and neighbour expansion."""

    async def get(self, chunk_ids: Sequence[str]) -> list[Chunk]: ...

    async def by_doc(self, doc_id: str) -> list[Chunk]: ...


@runtime_checkable
class VectorStore(DocStore, Protocol):
    async def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    async def delete_doc(self, doc_id: str) -> int: ...

    async def search(
        self,
        vector: np.ndarray,
        k: int,
        principal: Principal | None,
        flt: Filter | None = None,
    ) -> list[ScoredChunk]: ...

    async def count(self) -> int: ...


@runtime_checkable
class SparseIndex(Protocol):
    async def add(self, chunks: Sequence[Chunk]) -> None: ...

    async def delete_doc(self, doc_id: str) -> int: ...

    async def search(
        self, query: str, k: int, principal: Principal | None, flt: Filter | None = None
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class Retriever(Protocol):
    name: str

    async def retrieve(
        self, query: str, k: int, principal: Principal | None, flt: Filter | None = None
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...
