"""Post-retrieval expansion and priors (docs/07-retrieval.md §3).

* **Neighbour expansion** — add chunks ``i−w … i+w`` of top hits (fixes boundary cuts).
* **Small-to-big** — match children, return their parent chunks for generation.
* **Recency / authority boosting** — ``final = s + w_f · freshness + w_a · authority``.

Every chunk pulled in from the doc store is re-checked with
:func:`~chatbot.core.filters.visible`, so expansion can never leak a chunk the
principal could not have retrieved directly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from chatbot.core.filters import Filter, visible
from chatbot.core.interfaces import DocStore
from chatbot.core.types import Chunk, ChunkKind, Principal, ScoredChunk, utcnow


async def expand_neighbours(
    hits: Sequence[ScoredChunk],
    docstore: DocStore,
    principal: Principal | None,
    flt: Filter | None = None,
    *,
    window: int = 1,
    max_hits: int | None = None,
    score_decay: float = 0.9,
) -> list[ScoredChunk]:
    """Insert same-document neighbours (``position ± window``) right after each hit.

    Only the first ``max_hits`` hits are expanded. Neighbours get ``score × decay``
    and ``source="neighbour"``; chunks already present are not duplicated.
    """
    if window <= 0:
        return list(hits)
    present = {h.chunk.chunk_id for h in hits}
    doc_cache: dict[str, list[Chunk]] = {}
    out: list[ScoredChunk] = []
    expand_n = len(hits) if max_hits is None else max_hits
    for idx, h in enumerate(hits):
        out.append(h)
        if idx >= expand_n:
            continue
        c = h.chunk
        if c.doc_id not in doc_cache:
            doc_cache[c.doc_id] = await docstore.by_doc(c.doc_id)
        for n in doc_cache[c.doc_id]:
            if (
                n.chunk_id in present
                or n.kind != c.kind
                or n.doc_version != c.doc_version
                or abs(n.position - c.position) > window
                or not visible(n, principal, flt)
            ):
                continue
            present.add(n.chunk_id)
            s = h.score * score_decay
            out.append(
                ScoredChunk(chunk=n, score=s, source="neighbour", stage_scores={"neighbour": s})
            )
    return out


async def resolve_parents(
    hits: Sequence[ScoredChunk],
    docstore: DocStore,
    principal: Principal | None,
    flt: Filter | None = None,
    *,
    parent_kinds: frozenset[ChunkKind] = frozenset({ChunkKind.PARENT}),
) -> list[ScoredChunk]:
    """Small-to-big: replace children by their parent chunk (best child score wins).

    Hits without a resolvable/visible parent of a kind in ``parent_kinds`` are kept as-is.
    Order follows the best-scoring child of each parent.
    """
    ids = list(dict.fromkeys(h.chunk.parent_id for h in hits if h.chunk.parent_id))
    parents = {p.chunk_id: p for p in await docstore.get(ids)} if ids else {}
    out: dict[str, ScoredChunk] = {}
    for h in hits:
        p = parents.get(h.chunk.parent_id or "")
        if p is not None and p.kind in parent_kinds and visible(p, principal, flt):
            cand = ScoredChunk(
                chunk=p,
                score=h.score,
                source="parent",
                stage_scores={**h.stage_scores, "parent": h.score},
            )
        else:
            cand = h
        cur = out.get(cand.chunk.chunk_id)
        if cur is None or cand.score > cur.score:
            out[cand.chunk.chunk_id] = cand
    return sorted(out.values(), key=lambda s: -s.score)


def _parse_dt(v: object) -> datetime | None:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=UTC)
    if isinstance(v, str) and v:
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return None


def freshness_score(
    chunk: Chunk,
    *,
    now: datetime | None = None,
    half_life_days: float = 180.0,
    date_fields: Sequence[str] = ("updated_at", "effective_date", "created_at"),
) -> float:
    """Exponential decay in ``[0, 1]``: 1 for "now", 0.5 after ``half_life_days``; 0 if undated."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    now = now or utcnow()
    for f in date_fields:
        dt = _parse_dt(chunk.metadata.get(f))
        if dt is not None:
            age_days = max((now - dt).total_seconds() / 86400.0, 0.0)
            return math.exp(-math.log(2) * age_days / half_life_days)
    return 0.0


def authority_score(
    chunk: Chunk,
    *,
    field: str = "authority",
    by_doc_type: Mapping[str, float] | None = None,
    default: float = 0.0,
) -> float:
    """Authority prior in ``[0, 1]`` from ``metadata[field]`` or a doc-type table."""
    v = chunk.metadata.get(field)
    if isinstance(v, (int, float)):
        return float(min(max(v, 0.0), 1.0))
    if by_doc_type:
        dt = chunk.metadata.get("doc_type")
        if isinstance(dt, str) and dt in by_doc_type:
            return float(by_doc_type[dt])
    return default


def apply_boosts(
    hits: Sequence[ScoredChunk],
    *,
    freshness_weight: float = 0.0,
    authority_weight: float = 0.0,
    latest_weight: float = 0.0,
    now: datetime | None = None,
    half_life_days: float = 180.0,
    authority_by_doc_type: Mapping[str, float] | None = None,
    stage: str = "boosted",
) -> list[ScoredChunk]:
    """``final = s + w_f·freshness + w_a·authority + w_l·is_latest``; re-sorted descending."""
    now = now or utcnow()
    out: list[ScoredChunk] = []
    for h in hits:
        bonus = 0.0
        if freshness_weight:
            bonus += freshness_weight * freshness_score(
                h.chunk, now=now, half_life_days=half_life_days
            )
        if authority_weight:
            bonus += authority_weight * authority_score(h.chunk, by_doc_type=authority_by_doc_type)
        if latest_weight and h.chunk.metadata.get("is_latest", True):
            bonus += latest_weight
        out.append(h.with_score(h.score + bonus, stage))
    return sorted(out, key=lambda s: -s.score)
