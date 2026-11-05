"""Hybrid retrieval on a toy corpus: dense handles paraphrase, BM25 handles exact IDs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from chatbot.core.filters import Filter
from chatbot.core.types import ACL, Chunk, ChunkKind, Principal, ScoredChunk
from chatbot.embed import BaseEmbedder, HashingEmbedder
from chatbot.index import BM25Index, InMemoryVectorStore
from chatbot.retrieve import (
    DenseRetriever,
    FusionConfig,
    FusionMethod,
    HybridRetriever,
    MMRSelector,
    ParentMappingRetriever,
    SparseRetriever,
    apply_boosts,
    expand_neighbours,
    resolve_parents,
)

CONCEPTS = {
    "refund": 0,
    "money": 0,
    "reimburse": 0,
    "reimbursement": 0,
    "refunds": 0,
    "login": 1,
    "sign": 1,
    "sso": 1,
    "password": 1,
    "authentication": 1,
    "calendar": 2,
    "sync": 2,
    "schedule": 2,
    "error": 3,
    "fails": 3,
    "failure": 3,
    "broken": 3,
    "invoice": 4,
    "billing": 4,
    "bill": 4,
}


class ConceptEmbedder(BaseEmbedder):
    """Toy 'semantic' embedder: synonyms share a dimension; IDs are invisible to it."""

    def __init__(self) -> None:
        super().__init__(name="concept", dim=6, query_prefix="", document_prefix="")

    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace("-", " ").replace(".", " ").replace("?", " ").split():
                if w in CONCEPTS:
                    out[i, CONCEPTS[w]] += 1
            out[i, 5] += 0.05
        return out


DOCS = {
    "refund": "Refunds are issued to the original card within 5 days.",
    "sso": "Configure SSO authentication with Okta SAML.",
    "err4012": "ERR-4012: calendar sync error when the token expired.",
    "err5000": "ERR-5000: calendar sync error caused by rate limits.",
    "invoice": "Download invoices from the billing page.",
}
PUBLIC = ACL(public=True)
P = Principal(user_id="u", groups=frozenset({"staff"}))


def mk(cid: str, text: str, acl: ACL = PUBLIC, **kw: object) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=kw.pop("doc_id", cid), text=text, acl=acl, **kw)  # type: ignore[arg-type]


async def build(
    extra: Sequence[Chunk] = (),
) -> tuple[DenseRetriever, SparseRetriever, InMemoryVectorStore]:
    emb = ConceptEmbedder()
    store = InMemoryVectorStore(dim=emb.dim)
    bm25 = BM25Index()
    chunks = [mk(k, v) for k, v in DOCS.items()] + list(extra)
    await store.upsert(chunks, await emb.embed_documents([c.index_text for c in chunks]))
    await bm25.add(chunks)
    return DenseRetriever(emb, store), SparseRetriever(bm25), store


QUERIES = {
    "how do I get my money back": "refund",  # paraphrase: no lexical overlap
    "ERR-4012": "err4012",  # exact id: invisible to the concept embedder
    "calendar sync error ERR-5000": "err5000",
    "reimbursement for card": "refund",
}


async def recall_at_1(retriever: object) -> float:
    hits = 0
    for q, gold in QUERIES.items():
        res = await retriever.retrieve(q, 1, P)  # type: ignore[attr-defined]
        hits += bool(res) and res[0].chunk.chunk_id == gold
    return hits / len(QUERIES)


async def test_hybrid_beats_single_legs() -> None:
    dense, sparse, _ = await build()
    hybrid = HybridRetriever([dense, sparse], leg_k=10)
    r_dense, r_sparse, r_hybrid = (
        await recall_at_1(dense),
        await recall_at_1(sparse),
        await recall_at_1(hybrid),
    )
    assert r_hybrid == 1.0
    assert r_hybrid > r_dense and r_hybrid > r_sparse
    cfg = FusionConfig(method=FusionMethod.CONVEX, weights={"dense": 0.5, "bm25": 0.5})
    assert await recall_at_1(HybridRetriever([dense, sparse], fusion=cfg)) >= max(r_dense, r_sparse)
    assert set(hybrid.last_leg_latency_ms) == {"dense", "bm25"}


async def test_hybrid_same_acl_and_filter_on_all_legs() -> None:
    secret = mk(
        "secret", "ERR-4012 calendar sync error internal postmortem", ACL(allow_groups=["sre"])
    )
    dense, sparse, _ = await build([secret])
    hybrid = HybridRetriever([dense, sparse])
    hits = await hybrid.retrieve("ERR-4012 calendar sync error", 3, P)
    assert len(hits) == 3 and "secret" not in {h.chunk.chunk_id for h in hits}
    flt = Filter.eq(doc_id="err5000")
    assert {h.chunk.chunk_id for h in await hybrid.retrieve("calendar error", 5, P, flt)} == {
        "err5000"
    }


async def test_failed_leg_degrades_gracefully() -> None:
    _, sparse, _ = await build()

    class Broken:
        name = "broken"

        async def retrieve(self, *a: object, **k: object) -> list[ScoredChunk]:
            raise RuntimeError("down")

    hits = await HybridRetriever([sparse, Broken()]).retrieve("ERR-4012", 1, P)
    assert hits[0].chunk.chunk_id == "err4012"
    with pytest.raises(RuntimeError):
        await HybridRetriever([Broken()]).retrieve("x", 1, P)


async def test_question_leg_maps_to_parent_and_checks_acl() -> None:
    emb = HashingEmbedder(dim=128)
    qstore = InMemoryVectorStore(dim=128)
    docs = InMemoryVectorStore(dim=128)
    target = mk("t1", "Annual plans renew on the purchase date.")
    hidden = mk("t2", "Executive compensation bands.", ACL(allow_groups=["exec"]))
    await docs.upsert([target, hidden], await emb.embed_documents([target.text, hidden.text]))
    qs = [
        mk("q1", "When does my annual plan renew?", kind=ChunkKind.QUESTION, parent_id="t1"),
        mk("q2", "When does the annual exec plan renew?", kind=ChunkKind.QUESTION, parent_id="t2"),
    ]
    await qstore.upsert(qs, await emb.embed_documents([q.text for q in qs]))
    leg = ParentMappingRetriever(DenseRetriever(emb, qstore, name="questions"), docs)
    hits = await leg.retrieve("when does my annual plan renew", 5, P)
    assert [h.chunk.chunk_id for h in hits] == ["t1"]


async def test_mmr_diversity() -> None:
    emb = HashingEmbedder(dim=256)
    cands = [
        ScoredChunk(chunk=mk("a", "refund policy for annual plans"), score=0.9),
        ScoredChunk(chunk=mk("b", "refund policy for annual plans (copy)"), score=0.89),
        ScoredChunk(chunk=mk("c", "refund timeline and card processing"), score=0.5),
    ]
    picked = await MMRSelector(emb, lambda_=0.5).select("refund policy", cands, 2)
    assert [p.chunk.chunk_id for p in picked] == ["a", "c"]
    relevance_only = await MMRSelector(emb, lambda_=1.0).select(
        "refund policy for annual plans", cands, 2
    )
    assert {p.chunk.chunk_id for p in relevance_only} == {"a", "b"}


async def test_neighbour_parent_expansion_and_boosts() -> None:
    store = InMemoryVectorStore(dim=2)
    parent = mk("p", "whole section", kind=ChunkKind.PARENT, doc_id="d")
    kids = [mk(f"k{i}", f"part {i}", doc_id="d", position=i, parent_id="p") for i in range(4)]
    kids[3] = kids[3].model_copy(update={"acl": ACL(allow_groups=["x"])})
    await store.upsert([parent, *kids], np.ones((5, 2), dtype=np.float32))
    hit = ScoredChunk(chunk=kids[2], score=1.0)
    exp = await expand_neighbours([hit], store, P)
    assert [h.chunk.chunk_id for h in exp] == ["k2", "k1"]  # k3 is not visible
    parents = await resolve_parents([hit, ScoredChunk(chunk=kids[1], score=0.5)], store, P)
    assert [h.chunk.chunk_id for h in parents] == ["p"] and parents[0].score == 1.0

    new = ScoredChunk(
        chunk=mk("n", "x", metadata={"updated_at": "2026-11-01T00:00:00+00:00"}), score=0.50
    )
    old = ScoredChunk(
        chunk=mk("o", "x", metadata={"updated_at": "2020-01-01T00:00:00+00:00"}), score=0.52
    )
    boosted = apply_boosts([old, new], freshness_weight=0.1)
    assert boosted[0].chunk.chunk_id == "n" and "boosted" in boosted[0].stage_scores
