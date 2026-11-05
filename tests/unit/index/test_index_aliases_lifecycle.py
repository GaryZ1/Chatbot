from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from chatbot.core.types import ACL, Chunk, Document, Principal, utcnow
from chatbot.embed import HashingEmbedder
from chatbot.index import (
    AliasManager,
    BM25Index,
    IndexAction,
    IndexSchema,
    IndexWriter,
    InMemoryVectorStore,
)

P = Principal(user_id="u")


def chunks_for(doc: Document, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=Chunk.make_id(doc.doc_id, doc.version, i),
            doc_id=doc.doc_id,
            doc_version=doc.version,
            position=i,
            text=t,
            acl=ACL(public=True),
        )
        for i, t in enumerate(texts)
    ]


async def test_alias_switch_blue_green() -> None:
    emb = HashingEmbedder(dim=64)
    mgr = AliasManager(lambda schema: InMemoryVectorStore(dim=64, name=schema.collection_name))
    blue = IndexSchema(version=1)
    mgr.create_index(blue)
    live = mgr.view(blue.alias_name)
    with pytest.raises(KeyError):
        await live.count()
    await mgr.switch(blue.alias_name, blue.collection_name)
    doc = Document(doc_id="d", blocks=[])
    await mgr.backfill(blue.collection_name, chunks_for(doc, ["old text"]), emb)
    assert await live.count() == 1

    green = blue.next_version()
    validated: list[str] = []

    async def validate(store: object) -> bool:
        validated.append(green.collection_name)
        return True

    sw = await mgr.migrate(blue, green, chunks_for(doc, ["new a", "new b"]), emb, validate=validate)
    assert (sw.old, sw.new) == ("chunks_v1", "chunks_v2") and validated
    assert await live.count() == 2  # same proxy now reads green
    assert mgr.indexes() == ["chunks_v2"]
    with pytest.raises(ValueError):
        await mgr.drop("chunks_v2")  # still aliased


async def test_failed_validation_keeps_alias() -> None:
    mgr = AliasManager(lambda s: InMemoryVectorStore(dim=8))
    blue = IndexSchema()
    mgr.create_index(blue)
    await mgr.switch(blue.alias_name, blue.collection_name)

    async def bad(_: object) -> bool:
        return False

    with pytest.raises(RuntimeError):
        await mgr.migrate(blue, blue.next_version(), [], HashingEmbedder(dim=8), validate=bad)
    assert mgr.resolve(blue.alias_name) == "chunks_v1"


async def test_index_writer_lifecycle() -> None:
    emb = HashingEmbedder(dim=64)
    store, bm25 = InMemoryVectorStore(dim=64), BM25Index()
    w = IndexWriter(emb, store, bm25)
    doc = Document(doc_id="d", updated_at=utcnow() - timedelta(seconds=30))
    doc = doc.model_copy(update={"blocks": []})
    r = await w.index_document(doc, chunks_for(doc, ["alpha one", "beta two", "gamma three"]))
    assert r.action is IndexAction.INDEXED and r.chunks_written == 3
    assert r.freshness_lag_s is not None and r.freshness_lag_s >= 30
    assert (await w.index_document(doc, chunks_for(doc, ["x"]))).action is IndexAction.SKIPPED

    from chatbot.core.types import Block, BlockType

    doc2 = doc.model_copy(
        update={"version": "2", "blocks": [Block(type=BlockType.PARAGRAPH, text="changed")]}
    )
    r2 = await w.index_document(doc2, chunks_for(doc2, ["alpha changed"]))
    assert r2.action is IndexAction.INDEXED and r2.chunks_deleted == 3
    stored = await store.by_doc("d")
    assert [c.text for c in stored] == ["alpha changed"]
    assert (
        stored[0].metadata["is_latest"] is True
        and stored[0].metadata["embedding_model"] == "hashing@1:64"
    )
    assert [h.chunk.text for h in await bm25.search("beta", 5, P)] == []
    # A fresh writer recognises unchanged content from stored metadata.
    assert (
        await IndexWriter(emb, store, bm25).index_document(doc2, [])
    ).action is IndexAction.SKIPPED
    assert (
        await w.index_document(doc2.model_copy(update={"deleted": True}), [])
    ).action is IndexAction.DELETED
    assert await store.count() == 0 and await bm25.count() == 0
    assert w.freshness.summary()["count"] == 2.0


async def test_index_writer_keep_history() -> None:
    emb = HashingEmbedder(dim=32)
    store = InMemoryVectorStore(dim=32)
    w = IndexWriter(emb, store, keep_history=True)
    v1 = Document(doc_id="d", version="1")
    await w.index_document(v1, chunks_for(v1, ["first"]))
    from chatbot.core.types import Block, BlockType

    v2 = Document(doc_id="d", version="2", blocks=[Block(type=BlockType.PARAGRAPH, text="second")])
    await w.index_document(v2, chunks_for(v2, ["second"]))
    by_version = {c.doc_version: c.metadata["is_latest"] for c in await store.by_doc("d")}
    assert by_version == {"1": False, "2": True}
    from chatbot.core.filters import Filter

    hits = await store.search(await emb.embed_query("first"), 5, P, Filter.eq(is_latest=True))
    assert [h.chunk.doc_version for h in hits] == ["2"]
    assert np.isfinite(hits[0].score)
