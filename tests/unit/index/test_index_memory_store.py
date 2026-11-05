from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chatbot.core.filters import Filter
from chatbot.core.interfaces import DocStore, VectorStore
from chatbot.core.types import ACL, Chunk, Principal
from chatbot.index import InMemoryVectorStore

ALICE = Principal(user_id="alice", groups=frozenset({"eng"}))
PUBLIC = ACL(public=True)
SECRET = ACL(allow_groups=["hr"])


def chunk(doc: str, pos: int, acl: ACL = PUBLIC, tenant: str = "default", **meta: object) -> Chunk:
    return Chunk(
        chunk_id=f"{doc}-{pos}", doc_id=doc, position=pos, text=f"{doc} chunk {pos}",
        acl=acl, tenant_id=tenant, metadata=dict(meta),
    )


def vec(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


async def test_protocols() -> None:
    s = InMemoryVectorStore(dim=3)
    assert isinstance(s, VectorStore) and isinstance(s, DocStore)


async def test_acl_enforced_before_top_k() -> None:
    s = InMemoryVectorStore(dim=2)
    # The 5 restricted chunks are the *closest* to the query; 5 visible ones are farther.
    restricted = [chunk("secret", i, SECRET) for i in range(5)]
    visible_chunks = [chunk("pub", i) for i in range(5)]
    other_tenant = [chunk("t2", 0, tenant="acme")]
    await s.upsert(restricted, np.tile(vec(1, 0), (5, 1)))
    await s.upsert(visible_chunks, np.array([[1, 0.2 * (i + 1)] for i in range(5)], dtype=np.float32))
    await s.upsert(other_tenant, vec(1, 0))
    hits = await s.search(vec(1, 0), 3, ALICE)
    assert len(hits) == 3
    assert all(h.chunk.doc_id == "pub" for h in hits)
    assert [h.chunk.position for h in hits] == [0, 1, 2]
    assert hits[0].score >= hits[1].score >= hits[2].score
    # Trusted offline principal=None sees everything.
    assert (await s.search(vec(1, 0), 3, None))[0].chunk.doc_id in {"secret", "t2"}
    hr = Principal(user_id="bob", groups=frozenset({"hr"}))
    assert {h.chunk.doc_id for h in await s.search(vec(1, 0), 5, hr)} == {"secret"}


async def test_metadata_filter_and_upsert_delete() -> None:
    s = InMemoryVectorStore(dim=2)
    await s.upsert([chunk("a", 0, doc_type="faq"), chunk("b", 0, doc_type="policy")], np.eye(2, dtype=np.float32))
    hits = await s.search(vec(1, 0), 5, ALICE, Filter.eq(doc_type="policy"))
    assert [h.chunk.doc_id for h in hits] == ["b"]
    await s.upsert([chunk("a", 0, doc_type="faq")], vec(0, 1).reshape(1, -1))  # upsert by chunk_id
    assert await s.count() == 2
    assert (await s.search(vec(0, 1), 1, ALICE))[0].score == pytest.approx(1.0)
    assert await s.delete_doc("a") == 1
    assert await s.count() == 1 and await s.get(["a-0", "b-0"]) == [chunk("b", 0, doc_type="policy")]
    with pytest.raises(ValueError):
        await s.upsert([chunk("c", 0)], vec(1, 2, 3).reshape(1, -1))


async def test_by_doc_order_neighbours_and_persistence(tmp_path: Path) -> None:
    s = InMemoryVectorStore(dim=3)
    chunks = [chunk("d", p) for p in (3, 0, 2, 1, 4)]
    await s.upsert(chunks, np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32))
    assert [c.position for c in await s.by_doc("d")] == [0, 1, 2, 3, 4]
    assert [c.position for c in await s.neighbours("d-2", window=1)] == [1, 3]
    s.save(tmp_path / "store")
    loaded = InMemoryVectorStore.load(tmp_path / "store")
    assert await loaded.count() == 5
    q = vec(0.3, -1, 2)
    a = await s.search(q, 5, None)
    b = await loaded.search(q, 5, None)
    assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]
    assert [h.score for h in a] == pytest.approx([h.score for h in b])
    assert await loaded.get(["d-1"]) == await s.get(["d-1"])
