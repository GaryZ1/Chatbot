"""Dense (bi-encoder) retrieval leg (docs/07-retrieval.md §1).

The query is embedded with the embedder's *query* prefix (``embed_query``) — never
``embed_documents`` — and searched with the ACL/metadata pre-filter applied inside
the vector store.
"""

from __future__ import annotations

import logging

from chatbot.core.filters import Filter
from chatbot.core.interfaces import Embedder, VectorStore
from chatbot.core.types import Principal, ScoredChunk

log = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore, *, name: str = "dense") -> None:
        store_dim = getattr(store, "dim", None)
        if isinstance(store_dim, int) and store_dim != embedder.dim:
            raise ValueError(f"embedder dim {embedder.dim} != vector store dim {store_dim}")
        self.embedder = embedder
        self.store = store
        self.name = name

    async def retrieve(
        self, query: str, k: int, principal: Principal | None, flt: Filter | None = None
    ) -> list[ScoredChunk]:
        if k <= 0 or not query.strip():
            return []
        vec = await self.embedder.embed_query(query)
        hits = await self.store.search(vec, k, principal, flt)
        return [h.with_score(h.score, self.name) for h in hits]
