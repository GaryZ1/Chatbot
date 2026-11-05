from __future__ import annotations

from pathlib import Path

import pytest

from chatbot.core.interfaces import SparseIndex
from chatbot.core.types import ACL, Chunk, Principal
from chatbot.index import Analyzer, BM25Index, light_stem

P = Principal(user_id="u", groups=frozenset({"staff"}))


def c(cid: str, text: str, title: str = "", acl: ACL | None = None) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=cid.split("-")[0], text=text, title=title, acl=acl or ACL(public=True)
    )


CORPUS = [
    c("a-0", "Refunds for annual plans are prorated. Request a refund in the billing portal."),
    c("b-0", "Single sign-on: configure SAML with Okta or Azure AD."),
    c("c-0", "Error ERR-4012 means the calendar sync token expired; reconnect the calendar."),
    c("d-0", "Our office is closed on public holidays.", title="Refund policy"),
    c("e-0", "Annual plans renew automatically every year."),
]


def test_analyzer() -> None:
    an = Analyzer()
    assert an.analyze("The Refunds were processed") == ["refund", "process"]
    assert "err-4012" in an.analyze("see ERR-4012") and "4012" in an.analyze("see ERR-4012")
    assert light_stem("running") == "run" and light_stem("policies") == "policy"
    assert light_stem("class") == "class" and light_stem("v2s") == "v2s"


async def test_bm25_ranking_sanity() -> None:
    idx = BM25Index()
    assert isinstance(idx, SparseIndex)
    await idx.add(CORPUS)
    hits = await idx.search("how do I get a refund for annual plans", 3, P)
    assert hits[0].chunk.chunk_id == "a-0"
    assert hits[0].score > hits[1].score > 0
    ids = await idx.search("ERR-4012", 5, P)
    assert ids[0].chunk.chunk_id == "c-0"
    assert await idx.search("kubernetes", 5, P) == []
    # Title boost: a title-only match beats nothing but body matches beat it for multiterm queries
    t = await idx.search("refund policy", 5, P)
    assert t[0].chunk.chunk_id == "d-0"


async def test_bm25_acl_prefilter_and_incremental() -> None:
    idx = BM25Index()
    secret = c("s-0", "refund refund refund annual plans secret", acl=ACL(allow_groups=["finance"]))
    await idx.add([*CORPUS, secret])
    hits = await idx.search("refund annual plans", 2, P)
    assert len(hits) == 2 and "s-0" not in {h.chunk.chunk_id for h in hits}
    assert await idx.delete_doc("a") == 1
    assert "a-0" not in {h.chunk.chunk_id for h in await idx.search("refund", 5, P)}
    await idx.add([c("e-0", "now about refunds only")])  # replace by id
    assert await idx.count() == 5
    assert (await idx.search("renew automatically", 5, P)) == []


async def test_bm25_persistence(tmp_path: Path) -> None:
    idx = BM25Index(k1=1.5, title_boost=3.0)
    await idx.add(CORPUS)
    idx.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded.k1 == 1.5 and loaded.vocabulary_size == idx.vocabulary_size
    a = await idx.search("annual refund", 5, P)
    b = await loaded.search("annual refund", 5, P)
    assert [(h.chunk.chunk_id, round(h.score, 9)) for h in a] == [
        (h.chunk.chunk_id, round(h.score, 9)) for h in b
    ]
    with pytest.raises(ValueError):
        BM25Index.load(tmp_path, analyzer=Analyzer(stem=False))
