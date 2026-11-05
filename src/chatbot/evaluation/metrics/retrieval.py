"""Ranked-retrieval metrics (docs/12-evaluation.md §2, "Retrieval").

All functions take a *ranked* list of ids (doc or chunk level, best first) and the
relevant set — or a graded relevance map ``id -> grade`` for nDCG. Duplicates in
the ranking (e.g. several chunks of the same document mapped to doc ids) are
removed keeping the first occurrence, so doc-level metrics are not inflated.

Also provides the token-level IoU / recall / precision chunking metric from
Chroma's "Evaluating Chunking Strategies for Retrieval" (Smith & Troynikov, 2024),
which scores *how much* of the gold excerpt the retrieved chunks cover and how many
irrelevant tokens they drag in (docs/02-chunking.md, normalised by context tokens).
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TypeVar

from pydantic import BaseModel

from chatbot.core.tokens import Tokenizer

K = TypeVar("K", bound=Hashable)

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def dedupe(ranked: Iterable[K]) -> list[K]:
    """Remove duplicates keeping first (best) occurrence."""
    return list(dict.fromkeys(ranked))


def _rel_set(relevant: Iterable[str] | Mapping[str, int]) -> set[str]:
    if isinstance(relevant, Mapping):
        return {k for k, g in relevant.items() if g > 0}
    return set(relevant)


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str] | Mapping[str, int], k: int) -> float:
    """``|relevant ∩ top_k| / |relevant|``; 0.0 when there is nothing relevant."""
    _check_k(k)
    rel = _rel_set(relevant)
    if not rel:
        return 0.0
    return len(rel.intersection(dedupe(ranked)[:k])) / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str] | Mapping[str, int], k: int) -> float:
    """``|relevant ∩ top_k| / k`` (standard definition: denominator is k, not len(top_k))."""
    _check_k(k)
    rel = _rel_set(relevant)
    return len(rel.intersection(dedupe(ranked)[:k])) / k


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str] | Mapping[str, int], k: int) -> float:
    _check_k(k)
    rel = _rel_set(relevant)
    return 1.0 if rel.intersection(dedupe(ranked)[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str] | Mapping[str, int], k: int | None = None) -> float:
    """``1 / rank`` of the first relevant id (0 if none within ``k``). MRR = mean over queries."""
    rel = _rel_set(relevant)
    items = dedupe(ranked)
    if k is not None:
        _check_k(k)
        items = items[:k]
    for i, rid in enumerate(items, start=1):
        if rid in rel:
            return 1.0 / i
    return 0.0


def mrr(rankings: Sequence[Sequence[str]], relevants: Sequence[Iterable[str] | Mapping[str, int]]) -> float:
    if len(rankings) != len(relevants):
        raise ValueError("rankings and relevants must have equal length")
    if not rankings:
        return 0.0
    return sum(reciprocal_rank(r, g) for r, g in zip(rankings, relevants, strict=True)) / len(rankings)


def average_precision(ranked: Sequence[str], relevant: Iterable[str] | Mapping[str, int], k: int | None = None) -> float:
    rel = _rel_set(relevant)
    if not rel:
        return 0.0
    items = dedupe(ranked)[: k or None]
    hits, total = 0, 0.0
    for i, rid in enumerate(items, start=1):
        if rid in rel:
            hits += 1
            total += hits / i
    return total / (min(len(rel), k) if k else len(rel))


def dcg_at_k(gains: Sequence[float], k: int, exponential: bool = True) -> float:
    """DCG with the ``log2(rank + 1)`` discount; exponential gain ``2^g - 1`` by default."""
    _check_k(k)
    total = 0.0
    for i, g in enumerate(gains[:k], start=1):
        gain = (2.0**g - 1.0) if exponential else g
        total += gain / math.log2(i + 1)
    return total


def ndcg_at_k(
    ranked: Sequence[str],
    relevance: Iterable[str] | Mapping[str, int],
    k: int,
    exponential: bool = True,
) -> float:
    """Normalised DCG with graded relevance (a plain id set means binary grade 1)."""
    _check_k(k)
    grades: Mapping[str, int] = relevance if isinstance(relevance, Mapping) else dict.fromkeys(relevance, 1)
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)
    if not ideal:
        return 0.0
    gains = [float(grades.get(rid, 0)) for rid in dedupe(ranked)]
    idcg = dcg_at_k([float(g) for g in ideal], k, exponential)
    return dcg_at_k(gains, k, exponential) / idcg if idcg > 0 else 0.0


def retrieval_metrics(
    ranked: Sequence[str],
    relevance: Iterable[str] | Mapping[str, int],
    ks: Sequence[int] = DEFAULT_KS,
    prefix: str = "",
) -> dict[str, float]:
    """All standard metrics at each ``k`` as a flat dict (``recall@5`` ...)."""
    grades: Mapping[str, int] = relevance if isinstance(relevance, Mapping) else dict.fromkeys(relevance, 1)
    out: dict[str, float] = {}
    for k in ks:
        out[f"{prefix}recall@{k}"] = recall_at_k(ranked, grades, k)
        out[f"{prefix}precision@{k}"] = precision_at_k(ranked, grades, k)
        out[f"{prefix}hit@{k}"] = hit_at_k(ranked, grades, k)
        out[f"{prefix}ndcg@{k}"] = ndcg_at_k(ranked, grades, k)
    out[f"{prefix}mrr"] = reciprocal_rank(ranked, grades)
    return out


# --------------------------------------------------------------------------- token IoU (Chroma)

Span = tuple[str, int, int]  # (doc_id, char_start, char_end)


class TokenOverlap(BaseModel):
    iou: float
    recall: float
    precision: float
    relevant_tokens: int
    retrieved_tokens: int
    overlap_tokens: int


def _token_ids(
    spans: Sequence[Span],
    doc_texts: Mapping[str, str] | None,
    tokenizer: Tokenizer | None,
    token_cache: dict[str, list[tuple[int, int]]],
) -> list[set[tuple[str, int]]]:
    """Map each char span to the set of (doc_id, unit index) it covers.

    Units are tokens when a tokenizer and the document text are available, else
    characters (a faithful approximation since IoU is a ratio)."""
    out: list[set[tuple[str, int]]] = []
    for doc_id, start, end in spans:
        if end < start:
            raise ValueError(f"invalid span ({doc_id}, {start}, {end})")
        if tokenizer is not None and doc_texts is not None and doc_id in doc_texts:
            if doc_id not in token_cache:
                token_cache[doc_id] = tokenizer.spans(doc_texts[doc_id])
            units = {
                (doc_id, i) for i, (ts, te) in enumerate(token_cache[doc_id]) if ts < end and te > start
            }
        else:
            units = {(doc_id, i) for i in range(start, end)}
        out.append(units)
    return out


def token_iou(
    retrieved_spans: Sequence[Span],
    gold_spans: Sequence[Span],
    doc_texts: Mapping[str, str] | None = None,
    tokenizer: Tokenizer | None = None,
    count_redundant: bool = True,
) -> TokenOverlap:
    """Chroma's token-level IoU for one query.

    ``IoU = |t_e ∩ t_r| / (|t_e| + |t_r| - |t_e ∩ t_r|)`` where ``t_e`` are gold
    excerpt tokens and ``t_r`` retrieved tokens. With ``count_redundant`` (Chroma's
    definition) tokens retrieved several times by overlapping chunks count each time
    in ``|t_r|``, penalising overlap that wastes context budget.
    """
    cache: dict[str, list[tuple[int, int]]] = {}
    gold_units = set().union(*_token_ids(gold_spans, doc_texts, tokenizer, cache)) if gold_spans else set()
    ret_lists = _token_ids(retrieved_spans, doc_texts, tokenizer, cache)
    ret_union: set[tuple[str, int]] = set().union(*ret_lists) if ret_lists else set()
    n_ret = sum(len(s) for s in ret_lists) if count_redundant else len(ret_union)
    inter = len(gold_units & ret_union)
    denom = len(gold_units) + n_ret - inter
    return TokenOverlap(
        iou=inter / denom if denom > 0 else 0.0,
        recall=inter / len(gold_units) if gold_units else 0.0,
        precision=inter / n_ret if n_ret else 0.0,
        relevant_tokens=len(gold_units),
        retrieved_tokens=n_ret,
        overlap_tokens=inter,
    )
