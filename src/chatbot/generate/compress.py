"""Extractive context compression (docs/10-generation-context.md: "curate the smallest
high-signal set of tokens").

Keeps the sentences of each chunk that overlap the query most (IDF-weighted term
overlap), in their original order, within a token budget. No model calls: it is a
safe, deterministic pre-step before :class:`~chatbot.generate.context_builder.ContextBuilder`
when reranked chunks are long. Citations still map to the original chunk ids.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from chatbot.core.tokens import Tokenizer, default_tokenizer, truncate_to_tokens
from chatbot.core.types import ScoredChunk
from chatbot.generate.lexical import split_sentences, terms


class ExtractiveCompressor:
    """Query-focused sentence extraction.

    Args:
        min_overlap: sentences scoring below this fraction of the best sentence are dropped.
        keep_lead: always keep a chunk's first sentence (it usually carries the subject).
        drop_irrelevant: drop chunks with zero query overlap entirely.
    """

    def __init__(
        self,
        *,
        min_overlap: float = 0.15,
        keep_lead: bool = True,
        drop_irrelevant: bool = False,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.min_overlap = min_overlap
        self.keep_lead = keep_lead
        self.drop_irrelevant = drop_irrelevant
        self.tokenizer = tokenizer or default_tokenizer()

    def _scores(
        self, query_terms: set[str], sentences: list[str], idf: dict[str, float]
    ) -> list[float]:
        out = []
        for s in sentences:
            st = set(terms(s))
            out.append(sum(idf.get(t, 1.0) for t in st & query_terms))
        return out

    def compress_text(
        self, query: str, text: str, max_tokens: int, *, idf: dict[str, float] | None = None
    ) -> str:
        """Return the query-relevant extract of ``text`` (≤ ``max_tokens``), or ``""`` if irrelevant
        and ``drop_irrelevant`` is set."""
        if max_tokens <= 0:
            return ""
        sentences = split_sentences(text)
        if not sentences:
            return ""
        q = set(terms(query))
        scores = self._scores(q, sentences, idf or {})
        best = max(scores) if scores else 0.0
        if best <= 0:
            if self.drop_irrelevant:
                return ""
            return truncate_to_tokens(sentences[0], max_tokens, self.tokenizer)
        if self.tokenizer.count(text) <= max_tokens and all(
            s >= self.min_overlap * best for s in scores
        ):
            return text
        ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
        keep: set[int] = set()
        used = 0
        if self.keep_lead:
            lead_cost = self.tokenizer.count(sentences[0])
            if lead_cost <= max_tokens:
                keep.add(0)
                used += lead_cost
        for i in ranked:
            if i in keep or scores[i] < self.min_overlap * best:
                continue
            cost = self.tokenizer.count(sentences[i])
            if used + cost > max_tokens:
                continue
            keep.add(i)
            used += cost
        if not keep:
            return truncate_to_tokens(sentences[ranked[0]], max_tokens, self.tokenizer)
        return " ".join(sentences[i] for i in sorted(keep))

    def compress(
        self, query: str, chunks: Sequence[ScoredChunk], budget_tokens: int
    ) -> list[ScoredChunk]:
        """Compress ``chunks`` so their texts total at most ``budget_tokens``.

        Budget is shared evenly; unused budget of short chunks rolls over to later ones
        (chunks are processed in descending score order, output keeps input order).
        """
        if budget_tokens <= 0 or not chunks:
            return []
        all_sents = [set(terms(s)) for sc in chunks for s in split_sentences(sc.chunk.text)]
        df: Counter[str] = Counter(t for ss in all_sents for t in ss)
        n = max(1, len(all_sents))
        idf = {t: math.log(1 + n / c) for t, c in df.items()}

        order = sorted(range(len(chunks)), key=lambda i: -chunks[i].score)
        remaining = budget_tokens
        results: dict[int, ScoredChunk] = {}
        for pos, i in enumerate(order):
            share = remaining // max(1, len(order) - pos)
            sc = chunks[i]
            text = self.compress_text(query, sc.chunk.text, share, idf=idf)
            if not text:
                continue
            remaining -= self.tokenizer.count(text)
            if text == sc.chunk.text:
                results[i] = sc
                continue
            md = {
                **sc.chunk.metadata,
                "compressed": True,
                "original_token_count": self.tokenizer.count(sc.chunk.text),
            }
            chunk = sc.chunk.model_copy(update={"text": text, "metadata": md})
            results[i] = sc.model_copy(update={"chunk": chunk})
        return [results[i] for i in sorted(results)]
