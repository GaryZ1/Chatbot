"""Rank fusion (docs/07-retrieval.md §2).

* **RRF** — ``score(d) = Σ_i w_i / (k + rank_i(d))`` with ``k = 60`` and 1-based
  ranks; scale-free, robust, the production default.
* **Convex combination** — ``Σ_i α_i · norm_i(s_i(d))`` with per-query min-max or
  z-score normalisation (α ≈ 0.5–0.8 on dense, tuned on the eval set). A document
  missing from a leg gets that leg's minimum normalised score.

:func:`fuse_many` fuses result lists from multi-query / RAG-Fusion rewrites.
:func:`dedup` removes duplicates by ``chunk_id`` and by content hash (copies and
versions filling the top-K are a common bug, docs/07 §6).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from chatbot.core.types import ScoredChunk, sha256_text


class FusionMethod(StrEnum):
    RRF = "rrf"
    WEIGHTED_RRF = "weighted_rrf"
    CONVEX = "convex"


class Normalization(StrEnum):
    MINMAX = "minmax"
    ZSCORE = "zscore"


class FusionConfig(BaseModel):
    method: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60
    #: Leg name -> weight (weighted RRF / convex α). Missing legs default to 1.0.
    weights: dict[str, float] = Field(default_factory=dict)
    normalization: Normalization = Normalization.MINMAX

    @field_validator("rrf_k")
    @classmethod
    def _k_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("rrf_k must be >= 0")
        return v


def _weights(n: int, weights: Sequence[float] | None) -> list[float]:
    if weights is None:
        return [1.0] * n
    if len(weights) != n:
        raise ValueError(f"{len(weights)} weights for {n} result lists")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    return list(weights)


def _merge(acc: dict[str, ScoredChunk], hit: ScoredChunk) -> None:
    cur = acc.get(hit.chunk.chunk_id)
    if cur is None:
        acc[hit.chunk.chunk_id] = hit
        return
    merged = dict(cur.stage_scores)
    for stage, s in hit.stage_scores.items():
        merged[stage] = max(s, merged.get(stage, s))
    acc[hit.chunk.chunk_id] = ScoredChunk(
        chunk=cur.chunk, score=cur.score, source=cur.source, stage_scores=merged
    )


def _finish(
    acc: dict[str, ScoredChunk], fused: dict[str, float], stage: str, top_k: int | None
) -> list[ScoredChunk]:
    order = sorted(fused, key=lambda cid: (-fused[cid], cid))
    if top_k is not None:
        order = order[: max(top_k, 0)]
    return [acc[cid].with_score(fused[cid], stage) for cid in order]


def rrf(
    result_lists: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
    stage: str = "rrf",
) -> list[ScoredChunk]:
    """(Weighted) Reciprocal Rank Fusion. Duplicate ids within one list count once (best rank)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    ws = _weights(len(result_lists), weights)
    fused: dict[str, float] = {}
    acc: dict[str, ScoredChunk] = {}
    for w, hits in zip(ws, result_lists, strict=True):
        seen: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            cid = hit.chunk.chunk_id
            if cid in seen:
                continue
            seen.add(cid)
            fused[cid] = fused.get(cid, 0.0) + w / (k + rank)
            _merge(acc, hit)
    return _finish(acc, fused, stage, top_k)


def normalize_scores(
    scores: Sequence[float], method: Normalization | str = "minmax"
) -> list[float]:
    """Per-query normalisation of one leg's scores."""
    if not scores:
        return []
    m = Normalization(method)
    if m is Normalization.MINMAX:
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-12:
            return [1.0] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]
    if len(scores) < 2:
        return [0.0] * len(scores)
    mu = statistics.fmean(scores)
    sd = statistics.pstdev(scores)
    if sd < 1e-12:
        return [0.0] * len(scores)
    return [(s - mu) / sd for s in scores]


def convex_combination(
    result_lists: Sequence[Sequence[ScoredChunk]],
    *,
    weights: Sequence[float] | None = None,
    normalization: Normalization | str = "minmax",
    top_k: int | None = None,
    stage: str = "convex",
) -> list[ScoredChunk]:
    """``Σ α_i · norm(s_i)``; e.g. ``weights=[0.7, 0.3]`` for (dense, bm25)."""
    ws = _weights(len(result_lists), weights)
    acc: dict[str, ScoredChunk] = {}
    per_leg: list[tuple[float, dict[str, float]]] = []
    for hits in result_lists:
        best: dict[str, float] = {}
        for h in hits:
            cid = h.chunk.chunk_id
            best[cid] = max(h.score, best.get(cid, h.score))
            _merge(acc, h)
        ids = list(best)
        normed = dict(
            zip(ids, normalize_scores([best[i] for i in ids], normalization), strict=True)
        )
        floor = min(normed.values()) if normed else 0.0
        per_leg.append((floor, normed))
    fused = {
        cid: sum(w * leg.get(cid, floor) for w, (floor, leg) in zip(ws, per_leg, strict=True))
        for cid in acc
    }
    return _finish(acc, fused, stage, top_k)


def fuse(
    result_lists: Sequence[Sequence[ScoredChunk]],
    config: FusionConfig | None = None,
    *,
    names: Sequence[str] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Fuse named legs according to ``config`` (weights are looked up by leg name)."""
    cfg = config or FusionConfig()
    if names is not None and len(names) != len(result_lists):
        raise ValueError("names and result_lists differ in length")
    ws = [cfg.weights.get(n, 1.0) for n in names] if names is not None else None
    if cfg.method is FusionMethod.CONVEX:
        return convex_combination(
            result_lists, weights=ws, normalization=cfg.normalization, top_k=top_k
        )
    if cfg.method is FusionMethod.RRF:
        ws = None
    return rrf(result_lists, k=cfg.rrf_k, weights=ws, top_k=top_k)


def fuse_many(
    result_lists: Sequence[Sequence[ScoredChunk]],
    *,
    method: FusionMethod | str = FusionMethod.RRF,
    k: int = 60,
    top_k: int | None = None,
    dedup_content: bool = True,
) -> list[ScoredChunk]:
    """Multi-query / RAG-Fusion helper: fuse the results of several query variants."""
    cfg = FusionConfig(method=FusionMethod(method), rrf_k=k)
    fused = fuse(result_lists, cfg)
    if dedup_content:
        fused = dedup(fused)
    return fused[:top_k] if top_k is not None else fused


def content_key(hit: ScoredChunk) -> str:
    return sha256_text(hit.chunk.text)


def dedup(hits: Sequence[ScoredChunk], *, by_content: bool = True) -> list[ScoredChunk]:
    """Keep the first (best-ranked) hit per ``chunk_id`` and, optionally, per content hash."""
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    out: list[ScoredChunk] = []
    for h in hits:
        if h.chunk.chunk_id in seen_ids:
            continue
        if by_content:
            key = content_key(h)
            if key in seen_content:
                continue
            seen_content.add(key)
        seen_ids.add(h.chunk.chunk_id)
        out.append(h)
    return out
