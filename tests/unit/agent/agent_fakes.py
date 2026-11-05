"""Tiny in-memory Retriever / DocStore for agent tests (ACL-aware via visible())."""

from __future__ import annotations

import re
from collections.abc import Sequence

from chatbot.core.filters import Filter, visible
from chatbot.core.types import ACL, Chunk, Principal, ScoredChunk

_W = re.compile(r"\w+")


def tokens(text: str) -> set[str]:
    return {t.lower() for t in _W.findall(text)}


def make_chunk(
    cid: str, doc: str, text: str, pos: int = 0, heading: str = "", acl: ACL | None = None
) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc,
        text=text,
        position=pos,
        title=doc,
        heading_path=[heading] if heading else [],
        acl=acl or ACL(public=True),
    )


class FakeRetriever:
    name = "fake"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.calls: list[tuple[str, Principal | None]] = []

    async def retrieve(
        self, query: str, k: int, principal: Principal | None, flt: Filter | None = None
    ) -> list[ScoredChunk]:
        self.calls.append((query, principal))
        q = tokens(query)
        out = []
        for c in self.chunks:
            if not visible(c, principal, flt):
                continue
            score = len(q & tokens(c.text))
            if score:
                out.append(ScoredChunk(chunk=c, score=float(score), source=self.name))
        out.sort(key=lambda s: -s.score)
        return out[:k]


class FakeDocStore:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)

    async def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [c for c in self.chunks if c.chunk_id in set(chunk_ids)]

    async def by_doc(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]


SECRET = ACL(allow_groups=["security-team"])

CORPUS = [
    make_chunk(
        "r1",
        "file:refund-policy.md",
        "Customers can request a full refund within 30 days of purchase.",
        0,
        "Refund window",
    ),
    make_chunk(
        "r2",
        "file:refund-policy.md",
        "Annual plans cancelled after 30 days receive prorated credit.",
        1,
        "Annual plans",
    ),
    make_chunk(
        "p1",
        "file:pto-policy.md",
        "Full-time employees accrue 20 days of PTO per year.",
        0,
        "Accrual",
    ),
    make_chunk(
        "s1",
        "file:incident-reports.md",
        "Confidential incident report: breach in March.",
        0,
        "Incidents",
        acl=SECRET,
    ),
]
