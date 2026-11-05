"""Reranker base class and shared text handling (docs/08-reranking.md §3).

Design rules implemented here:

* rerank on the **embed text with breadcrumbs/context** but return the original
  chunk (citation text is untouched) — :func:`rerank_text`;
* respect the model's max length by **truncating intelligently**: keep the heading
  path and the most query-relevant window of the body — :func:`smart_truncate`;
* the score of every stage is kept in ``ScoredChunk.stage_scores``.
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Sequence

from chatbot.core.tokens import Tokenizer, default_tokenizer
from chatbot.core.types import Chunk, ScoredChunk
from chatbot.index.analyzer import DEFAULT_ANALYZER, Analyzer

log = logging.getLogger(__name__)


def heading_line(chunk: Chunk) -> str:
    path = [p for p in chunk.heading_path if p]
    if chunk.title and (not path or path[0] != chunk.title):
        path = [chunk.title, *path]
    return " > ".join(path)


def rerank_text(chunk: Chunk) -> str:
    """Text the reranker sees: contextual ``embed_text`` (or heading line + text)."""
    if chunk.embed_text:
        return chunk.embed_text
    head = heading_line(chunk)
    return f"{head}\n{chunk.text}" if head else chunk.text


def smart_truncate(
    query: str,
    chunk: Chunk,
    max_tokens: int,
    *,
    tokenizer: Tokenizer | None = None,
    analyzer: Analyzer | None = None,
) -> str:
    """Fit ``rerank_text(chunk)`` into ``max_tokens``: heading + best query window.

    The window is the contiguous token span of the body with the most query-term
    matches (ties → earliest), so long chunks are not judged on their first 512
    tokens only.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    tok = tokenizer or default_tokenizer()
    an = analyzer or DEFAULT_ANALYZER
    full = rerank_text(chunk)
    if tok.count(full) <= max_tokens:
        return full
    head = heading_line(chunk)
    body = chunk.embed_text or chunk.text
    if head and body.startswith(head):
        body = body[len(head) :].lstrip()
    head_tokens = tok.count(head) if head else 0
    if head_tokens >= max_tokens // 2:  # pathological heading: keep the body instead
        head, head_tokens = "", 0
    budget = max(max_tokens - head_tokens - (1 if head else 0), 1)
    spans = tok.spans(body)
    if len(spans) <= budget:
        window = body
    else:
        qterms = set(an.analyze(query))
        hits = [1 if qterms.intersection(an.analyze(body[s:e])) else 0 for s, e in spans]
        cur = sum(hits[:budget])
        best_start, best = 0, cur
        for start in range(1, len(spans) - budget + 1):
            cur += hits[start + budget - 1] - hits[start - 1]
            if cur > best:
                best, best_start = cur, start
        s0 = spans[best_start][0]
        e0 = spans[best_start + budget - 1][1]
        window = body[s0:e0]
    return f"{head}\n{window}" if head else window


class BaseReranker(abc.ABC):
    """Template implementing :class:`chatbot.core.interfaces.Reranker`."""

    name: str = "reranker"

    @abc.abstractmethod
    async def score(self, query: str, candidates: Sequence[ScoredChunk]) -> list[float]:
        """One relevance score per candidate (higher is better)."""

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if top_k <= 0 or not candidates:
            return []
        scores = await self.score(query, candidates)
        if len(scores) != len(candidates):
            raise ValueError(f"{self.name} returned {len(scores)} scores for {len(candidates)} candidates")
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return [candidates[i].with_score(float(scores[i]), self.name) for i in order[:top_k]]
