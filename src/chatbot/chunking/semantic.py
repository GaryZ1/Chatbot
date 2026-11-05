"""Semantic (embedding-breakpoint) chunking (docs/02-chunking.md §2.9).

Each sentence is embedded together with ``buffer_size`` neighbours on each side;
the cosine distance between consecutive windows is computed and the document is
split where the distance exceeds the ``breakpoint_percentile`` (90–95). Min/max
token guards keep sizes sane; tables and code are hard boundaries and are never
split.

Evidence (Qu et al. 2024) shows no consistent gain over recursive splitting —
only adopt it after it wins on *your* evaluation.
"""

from __future__ import annotations

import logging

import numpy as np

from chatbot.chunking.base import BaseChunker, DocIndex, run_sync
from chatbot.chunking.recursive import RecursiveSplitter
from chatbot.chunking.sentence import Segmenter, split_sentences
from chatbot.core.interfaces import Embedder
from chatbot.core.tokens import Tokenizer
from chatbot.core.types import Chunk, Document

logger = logging.getLogger(__name__)


def cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """Distances ``1 - cos(v_i, v_{i+1})`` between consecutive rows."""
    if len(vectors) < 2:
        return np.zeros(0, dtype=np.float32)
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.where(norms == 0, 1.0, norms)
    return 1.0 - np.sum(v[:-1] * v[1:], axis=1)


class SemanticChunker(BaseChunker):
    """Async chunker: requires an :class:`~chatbot.core.interfaces.Embedder`.

    Args:
        embedder: sentence-window embedder (one call per document, batched).
        breakpoint_percentile: distance percentile above which to split (90–95).
        buffer_size: neighbouring sentences on each side included in each window (1–3).
        min_tokens / max_tokens: size guards.
    """

    name = "semantic"

    def __init__(
        self,
        embedder: Embedder,
        breakpoint_percentile: float = 95.0,
        buffer_size: int = 1,
        min_tokens: int = 50,
        max_tokens: int = 512,
        tokenizer: Tokenizer | None = None,
        *,
        segmenter: Segmenter = split_sentences,
        breadcrumbs: bool = True,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=breadcrumbs)
        self._validate_sizes(max_tokens)
        if not 0 < breakpoint_percentile < 100:
            raise ValueError("breakpoint_percentile must be in (0, 100)")
        if buffer_size < 0:
            raise ValueError("buffer_size must be >= 0")
        if not 0 <= min_tokens < max_tokens:
            raise ValueError("min_tokens must be in [0, max_tokens)")
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.segmenter = segmenter

    def chunk(self, doc: Document) -> list[Chunk]:
        result: list[Chunk] = run_sync(lambda: self.achunk(doc))
        return result

    async def achunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        text = index.text
        # segments: (start, end, atomic)
        segs: list[tuple[int, int, bool]] = []
        for span in index.spans:
            if span.atomic:
                segs.append((span.start, span.end, True))
            else:
                segs.extend((s, e, False) for s, e in self.segmenter(text, span.start, span.end))
        if not segs:
            return []
        prose = [i for i, sg in enumerate(segs) if not sg[2]]
        breaks: set[int] = set()  # break *after* segment i
        if len(prose) >= 2:
            windows = []
            for k in range(len(prose)):
                lo = prose[max(0, k - self.buffer_size)]
                hi = prose[min(len(prose) - 1, k + self.buffer_size)]
                windows.append(text[segs[lo][0] : segs[hi][1]])
            vecs = await self.embedder.embed_documents(windows)
            dists = cosine_distances(np.asarray(vecs))
            threshold = float(np.percentile(dists, self.breakpoint_percentile))
            breaks = {prose[k] for k, d in enumerate(dists) if d > threshold}
            logger.debug("semantic: %d sentences, threshold %.4f, %d breaks", len(prose), threshold, len(breaks))
        splitter = RecursiveSplitter(self.max_tokens, tokenizer=self.tokenizer)
        ranges: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = None
        for i, (s, e, atomic) in enumerate(segs):
            if atomic:
                if cur is not None:
                    ranges.append(cur)
                    cur = None
                ranges.append((s, e))
                continue
            if self.count(text[s:e]) > self.max_tokens:
                if cur is not None:
                    ranges.append(cur)
                    cur = None
                ranges.extend(splitter.split(text, s, e))
                continue
            if cur is None:
                cur = (s, e)
            elif self.count(text[cur[0] : e]) > self.max_tokens:
                ranges.append(cur)
                cur = (s, e)
            else:
                cur = (cur[0], e)
            if i in breaks and self.count(text[cur[0] : cur[1]]) >= self.min_tokens:
                ranges.append(cur)
                cur = None
        if cur is not None:
            if ranges and self.count(text[cur[0] : cur[1]]) < self.min_tokens and self.count(
                text[ranges[-1][0] : cur[1]]
            ) <= self.max_tokens and not index.span_at(ranges[-1][0]).atomic:
                ranges[-1] = (ranges[-1][0], cur[1])
            else:
                ranges.append(cur)
        return [
            self.make_chunk(index, position=i, start=s, end=e, metadata={"semantic_breaks": len(breaks)})
            for i, (s, e) in enumerate(ranges)
        ]
