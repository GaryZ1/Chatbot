"""Context-level post-processing after reranking (docs/08-reranking.md §4).

* **Dedup** near-identical chunks (versions, copies): exact content hash, then
  word-shingle Jaccard similarity.
* **Ordering**: "lost in the middle" (Liu et al. 2023) — most relevant first *and*
  last; or document order for narrative coherence.
* **Grouping** by document with a source header for clean citations.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from chatbot.core.types import ScoredChunk, sha256_text

_WORD = re.compile(r"\w+", re.UNICODE)


def _shingles(text: str, n: int) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.casefold())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def dedup_near_identical(
    hits: Sequence[ScoredChunk], *, threshold: float = 0.85, shingle: int = 3
) -> list[ScoredChunk]:
    """Drop hits whose text is (near-)identical to a better-ranked kept hit."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    kept: list[ScoredChunk] = []
    kept_sh: list[set[tuple[str, ...]]] = []
    hashes: set[str] = set()
    for h in hits:
        hsh = sha256_text(h.chunk.text)
        if hsh in hashes:
            continue
        sh = _shingles(h.chunk.text, shingle)
        if any(jaccard(sh, other) >= threshold for other in kept_sh):
            continue
        hashes.add(hsh)
        kept.append(h)
        kept_sh.append(sh)
    return kept


def lost_in_the_middle_order(hits: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Best first and second-best last, weakest in the middle: ranks 1,3,5,…,6,4,2."""
    ranked = sorted(hits, key=lambda h: -h.score)
    front = ranked[0::2]
    back = ranked[1::2]
    return [*front, *reversed(back)]


def document_order(hits: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Documents by best score; chunks inside a document by position (narrative order)."""
    return [c for g in group_by_document(hits) for c in g.chunks]


@dataclass
class DocumentGroup:
    doc_id: str
    title: str
    source_uri: str
    score: float
    chunks: list[ScoredChunk] = field(default_factory=list)

    @property
    def header(self) -> str:
        src = f" ({self.source_uri})" if self.source_uri else ""
        return f"{self.title or self.doc_id}{src}"


def group_by_document(hits: Sequence[ScoredChunk]) -> list[DocumentGroup]:
    groups: dict[str, DocumentGroup] = {}
    for h in hits:
        c = h.chunk
        g = groups.get(c.doc_id)
        if g is None:
            g = groups[c.doc_id] = DocumentGroup(c.doc_id, c.title, c.source_uri, h.score)
        g.score = max(g.score, h.score)
        g.chunks.append(h)
    for g in groups.values():
        g.chunks.sort(key=lambda s: (s.chunk.page or 0, s.chunk.position))
    return sorted(groups.values(), key=lambda g: -g.score)


class ContextOrder(StrEnum):
    SCORE = "score"
    LOST_IN_THE_MIDDLE = "lost_in_the_middle"
    DOCUMENT = "document"


def postprocess_context(
    hits: Sequence[ScoredChunk],
    *,
    order: ContextOrder | str = ContextOrder.LOST_IN_THE_MIDDLE,
    dedup_threshold: float | None = 0.85,
) -> list[ScoredChunk]:
    """Dedup then order the final context chunks."""
    out = dedup_near_identical(hits, threshold=dedup_threshold) if dedup_threshold else list(hits)
    match ContextOrder(order):
        case ContextOrder.SCORE:
            return sorted(out, key=lambda h: -h.score)
        case ContextOrder.LOST_IN_THE_MIDDLE:
            return lost_in_the_middle_order(out)
        case ContextOrder.DOCUMENT:
            return document_order(out)
