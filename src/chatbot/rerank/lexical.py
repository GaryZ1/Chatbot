"""Dependency-free lexical reranker (default and test reranker; docs/08 §1 "rule" tier).

Scores each candidate against the query using the shared BM25 analyzer:

``score = 0.6 · coverage + 0.3 · bm25 / (bm25 + 1) + 0.1 · phrase``

* ``coverage`` — IDF-weighted fraction of query terms present (rewards answering
  *all* parts of the question);
* ``bm25`` — BM25 over the candidate set as the corpus (k1=1.2, b=0.75);
* ``phrase`` — 1 if the normalised query appears verbatim.

Scores lie in ``[0, 1]``, which makes threshold calibration (``calibration.py``) easy.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from chatbot.core.types import ScoredChunk
from chatbot.index.analyzer import DEFAULT_ANALYZER, Analyzer
from chatbot.rerank.base import BaseReranker, rerank_text


class LexicalOverlapReranker(BaseReranker):
    def __init__(
        self,
        *,
        analyzer: Analyzer | None = None,
        k1: float = 1.2,
        b: float = 0.75,
        weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
        name: str = "lexical",
    ) -> None:
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("weights must be non-negative with a positive sum")
        total = sum(weights)
        self.weights = (weights[0] / total, weights[1] / total, weights[2] / total)
        self.analyzer = analyzer or DEFAULT_ANALYZER
        self.k1 = k1
        self.b = b
        self.name = name

    async def score(self, query: str, candidates: Sequence[ScoredChunk]) -> list[float]:
        qterms = list(dict.fromkeys(self.analyzer.analyze(query)))
        if not qterms:
            return [0.0] * len(candidates)
        docs = [Counter(self.analyzer.analyze(rerank_text(c.chunk))) for c in candidates]
        n = len(docs)
        avg_len = max(sum(sum(d.values()) for d in docs) / max(n, 1), 1e-9)
        df = {t: sum(1 for d in docs if t in d) for t in qterms}
        # Smoothed IDF over the candidate set; floor keeps common-but-required terms meaningful.
        idf = {t: max(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)), 0.1) for t in qterms}
        idf_total = sum(idf.values())
        qnorm = " ".join(query.casefold().split())
        w_cov, w_bm25, w_phrase = self.weights
        out: list[float] = []
        for cand, d in zip(candidates, docs, strict=True):
            length = sum(d.values())
            coverage = sum(idf[t] for t in qterms if t in d) / idf_total
            bm25 = 0.0
            for t in qterms:
                tf = d.get(t, 0)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * length / avg_len)
                    bm25 += idf[t] * tf * (self.k1 + 1) / denom
            phrase = 1.0 if len(qterms) > 1 and qnorm in " ".join(rerank_text(cand.chunk).casefold().split()) else 0.0
            out.append(w_cov * coverage + w_bm25 * bm25 / (bm25 + 1.0) + w_phrase * phrase)
        return out
