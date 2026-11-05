"""Choosing K (docs/07-retrieval.md §4).

After reranking, cut by **token budget** rather than a magic number, and cut where
scores **drop sharply** or fall **below a calibrated threshold** — which is what
enables principled abstention (an empty result).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from chatbot.core.tokens import Tokenizer, default_tokenizer
from chatbot.core.types import ScoredChunk


def score_gap_cutoff(
    hits: Sequence[ScoredChunk],
    *,
    min_k: int = 1,
    max_k: int = 20,
    threshold: float | None = None,
    rel_gap: float = 0.5,
    abs_gap: float | None = None,
) -> list[ScoredChunk]:
    """Dynamic K over score-sorted hits.

    * drop hits scoring below ``threshold`` (may return ``[]`` → abstain), even below ``min_k``;
    * after ``min_k`` hits, stop at the first drop ``s_i − s_{i+1}`` larger than
      ``rel_gap × (s_0 − s_last)`` (relative to the observed score range) or ``abs_gap``;
    * never return more than ``max_k``.
    """
    if min_k < 0 or max_k < 0:
        raise ValueError("min_k and max_k must be >= 0")
    ranked = sorted(hits, key=lambda h: -h.score)
    if threshold is not None:
        ranked = [h for h in ranked if h.score >= threshold]
    if not ranked:
        return []
    # The gap scale is the score range of the whole candidate list, not of the cut.
    span = ranked[0].score - ranked[-1].score
    ranked = ranked[:max_k]
    if len(ranked) <= max(min_k, 1):
        return ranked
    out = [ranked[0]]
    for prev, cur in itertools.pairwise(ranked):
        gap = prev.score - cur.score
        if len(out) >= min_k and (
            (abs_gap is not None and gap > abs_gap) or (span > 0 and gap > rel_gap * span)
        ):
            break
        out.append(cur)
    return out


def token_budget_cutoff(
    hits: Sequence[ScoredChunk],
    max_tokens: int,
    *,
    tokenizer: Tokenizer | None = None,
    min_k: int = 1,
) -> list[ScoredChunk]:
    """Keep hits in order while their total ``text`` tokens fit ``max_tokens``.

    Uses ``chunk.token_count`` when set. Always keeps ``min_k`` hits (they will be
    truncated downstream if needed).
    """
    tok = tokenizer or default_tokenizer()
    out: list[ScoredChunk] = []
    used = 0
    for h in hits:
        n = h.chunk.token_count or tok.count(h.chunk.text)
        if used + n > max_tokens and len(out) >= min_k:
            break
        out.append(h)
        used += n
    return out
