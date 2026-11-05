import numpy as np

from chatbot.core.types import Answer, AnswerStatus, Citation, Principal
from chatbot.ops.semantic_cache import SemanticCache, normalize_query

ALICE = Principal(user_id="alice", groups=frozenset({"eng"}))
BOB = Principal(user_id="bob", groups=frozenset({"sales"}))


class VecEmbedder:
    """Maps known queries to fixed vectors; unknown queries to a random-ish vector."""

    name = "vec"
    dim = 3

    def __init__(self, table):
        self.table = table

    async def embed_documents(self, texts):
        return np.stack([await self.embed_query(t) for t in texts])

    async def embed_query(self, text):
        return np.asarray(self.table.get(text, [0.0, 0.0, 1.0]), dtype=np.float32)


def answer(doc="d1"):
    return Answer(
        text="Use the VPN portal.", citations=[Citation(index=1, chunk_id="c", doc_id=doc)]
    )


def test_normalize_query():
    assert normalize_query("  How do I reset my PASSWORD?? ") == "how do i reset my password"


async def test_exact_hit_and_scope_isolation():
    cache = SemanticCache()
    assert await cache.put("How do I reset my password?", ALICE, "v1", answer())
    hit = await cache.get("how do i reset my password", ALICE, "v1")
    assert hit is not None and hit.kind == "exact"
    assert await cache.get("How do I reset my password?", BOB, "v1") is None  # never across scopes
    assert await cache.get("How do I reset my password?", ALICE, "v2") is None  # index version


async def test_semantic_hit_threshold_and_scope():
    emb = VecEmbedder(
        {
            "reset password": [1.0, 0.0, 0.0],
            "password reset steps": [0.99, 0.05, 0.0],  # cos ~0.9987
            "vpn setup": [0.7, 0.7, 0.0],  # cos ~0.707
        }
    )
    cache = SemanticCache(emb, threshold=0.95)
    await cache.put("reset password", ALICE, "v1", answer())
    hit = await cache.get("password reset steps", ALICE, "v1")
    assert hit is not None and hit.kind == "semantic" and hit.similarity > 0.95
    assert await cache.get("vpn setup", ALICE, "v1") is None
    assert await cache.get("password reset steps", BOB, "v1") is None
    # same groups but different user is a different scope too
    other_eng = Principal(user_id="carol", groups=frozenset({"eng"}))
    assert await cache.get("reset password", other_eng, "v1") is None


async def test_invalidate_on_doc_update_and_version():
    cache = SemanticCache()
    await cache.put("q1", ALICE, "v1", answer("d1"), doc_ids=["d1", "d2"])
    await cache.put("q2", BOB, "v1", answer("d3"), doc_ids=["d3"])
    assert await cache.invalidate_doc("d2") == 1
    assert await cache.get("q1", ALICE, "v1") is None
    assert await cache.get("q2", BOB, "v1") is not None
    assert await cache.invalidate_index_version(keep="v2") == 1
    assert len(cache) == 0


async def test_ttl_and_uncacheable():
    now = [0.0]
    cache = SemanticCache(ttl_s=10, clock=lambda: now[0])
    await cache.put("q", ALICE, "v1", answer())
    now[0] = 11
    assert await cache.get("q", ALICE, "v1") is None
    assert not await cache.put("q", ALICE, "v1", Answer(text="?", status=AnswerStatus.ABSTAINED))
    assert not await cache.put("q", ALICE, "v1", Answer(text="no docs"))
    assert cache.stats.rejected == {"status:abstained": 1, "no_documents": 1}
