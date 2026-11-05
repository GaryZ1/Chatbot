from __future__ import annotations

import numpy as np
import pytest

from chatbot.core.interfaces import Embedder
from chatbot.embed import (
    BaseEmbedder,
    CachingEmbedder,
    HashingEmbedder,
    QueryEmbeddingCache,
    build_embedder,
    default_prefixes,
)


class RecordingEmbedder(BaseEmbedder):
    def __init__(self, name: str = "toy", **kw: object) -> None:
        super().__init__(name=name, dim=4, **kw)  # type: ignore[arg-type]
        self.seen: list[tuple[list[str], bool]] = []

    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        self.seen.append((list(texts), is_query))
        return np.tile(np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


async def test_hashing_is_deterministic_normalised_and_protocol() -> None:
    e = HashingEmbedder(dim=256)
    assert isinstance(e, Embedder)
    a = await e.embed_documents(["The refund policy for annual plans", "x"])
    b = await e.embed_documents(["The refund policy for annual plans"])
    assert a.shape == (2, 256) and a.dtype == np.float32
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert (await e.embed_documents([])).shape == (0, 256)


async def test_hashing_similarity_tracks_overlap() -> None:
    e = HashingEmbedder()
    q = await e.embed_query("how do refunds work for annual plans")
    docs = await e.embed_documents(
        [
            "Refunds for annual plans are prorated and processed in 5 days.",
            "Configure SSO with Okta using SAML metadata.",
        ]
    )
    assert cos(q, docs[0]) > cos(q, docs[1]) + 0.2
    # morphology: char n-grams make "refund" ~ "refunds"
    r1, r2, r3 = await e.embed_documents(["refund", "refunds", "kubernetes"])
    assert cos(r1, r2) > 0.5 > cos(r1, r3)


async def test_query_prefix_is_applied_and_known_defaults() -> None:
    e = RecordingEmbedder(query_prefix="query: ", document_prefix="passage: ", batch_size=2)
    v = await e.embed_query("hello")
    assert e.seen[-1] == (["query: hello"], True)
    np.testing.assert_allclose(v, [0.6, 0.8, 0, 0], atol=1e-6)
    await e.embed_documents(["a", "b", "c"])
    assert e.seen[1:] == [(["passage: a", "passage: b"], False), (["passage: c"], False)]

    e5 = RecordingEmbedder(name="intfloat/e5-large-v2")
    assert e5.query_prefix == "query: " and e5.document_prefix == "passage: "
    assert default_prefixes("BAAI/bge-base-en-v1.5")[0].startswith("Represent")  # type: ignore[index]
    assert RecordingEmbedder(name="intfloat/e5-large-v2", query_prefix="").query_prefix == ""
    assert e5.info.key == "intfloat/e5-large-v2@1:4"


async def test_shape_mismatch_raises() -> None:
    class Bad(RecordingEmbedder):
        async def _embed_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
            return np.zeros((len(texts), 3), dtype=np.float32)

    with pytest.raises(ValueError):
        await Bad().embed_query("x")


async def test_caching_embedder_hits_on_normalised_text() -> None:
    inner = RecordingEmbedder()
    cache = QueryEmbeddingCache(maxsize=2)
    ce = CachingEmbedder(inner, cache)
    a = await ce.embed_query("hello   world")
    b = await ce.embed_query(" hello world ")
    np.testing.assert_allclose(a, b)
    assert len(inner.seen) == 1 and cache.stats.hits == 1
    await ce.embed_query("two")
    await ce.embed_query("three")  # evicts "hello world"
    await ce.embed_query("hello world")
    assert len(inner.seen) == 4
    assert ce.dim == 4


def test_factory() -> None:
    e = build_embedder("hashing", dim=128)
    assert isinstance(e, HashingEmbedder) and e.dim == 128
    with pytest.raises(ValueError):
        build_embedder("nope")
    with pytest.raises(ValueError):
        build_embedder("hashing:model-x")
