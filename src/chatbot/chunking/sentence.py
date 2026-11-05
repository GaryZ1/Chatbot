"""Sentence, paragraph, sliding-window and page chunking (docs/02-chunking.md §2.3, §2.4,
§2.7, §2.8).

The sentence segmenter is a dependency-free rule-based splitter (abbreviations,
initials, decimals, list items, line breaks). Swap in pySBD/spaCy by passing a
``segmenter`` callable returning ``(start, end)`` spans.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from chatbot.chunking.base import BaseChunker, DocIndex, trim_span
from chatbot.chunking.recursive import RecursiveSplitter
from chatbot.core.tokens import Tokenizer
from chatbot.core.types import Chunk, Document

Segmenter = Callable[[str, int, int], list[tuple[int, int]]]

ABBREVIATIONS: frozenset[str] = frozenset(
    "mr mrs ms dr prof sr jr st vs etc e.g i.e cf al fig figs no nos vol pp approx inc ltd "
    "co corp dept est jan feb mar apr jun jul aug sep sept oct nov dec mt ave u.s u.k "
    "z.b bzw usw ca resp".split()
)
_BOUNDARY = re.compile(r"[.!?…。！？]+['\"’”)\]]*(?=\s)|\n")


def split_sentences(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Rule-based sentence spans within ``text[start:end]`` (trimmed, offsets absolute)."""
    end = len(text) if end is None else end
    spans: list[tuple[int, int]] = []
    cur = start
    for m in _BOUNDARY.finditer(text, start, end):
        if m.group() != "\n":
            j = m.start()
            word_start = j
            while word_start > cur and not text[word_start - 1].isspace():
                word_start -= 1
            word = text[word_start:j].lower().strip("([\"'")
            if word in ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                continue
            nxt = m.end()
            while nxt < end and text[nxt] in " \t":
                nxt += 1
            if nxt < end and text[nxt].islower():
                continue
        s, e = trim_span(text, cur, m.end())
        if s < e:
            spans.append((s, e))
        cur = m.end()
    s, e = trim_span(text, cur, end)
    if s < e:
        spans.append((s, e))
    return spans


def _block_sentences(index: DocIndex, segmenter: Segmenter) -> list[tuple[int, int]]:
    """Sentences per block; tables and code are single, unsplittable units."""
    out: list[tuple[int, int]] = []
    for span in index.spans:
        if span.atomic:
            out.append((span.start, span.end))
        else:
            out.extend(segmenter(index.text, span.start, span.end))
    return out


class SentenceChunker(BaseChunker):
    """Pack whole sentences up to ``max_tokens`` (or exactly ``sentences_per_chunk``)."""

    name = "sentence"

    def __init__(
        self,
        max_tokens: int = 256,
        sentences_per_chunk: int | None = None,
        overlap_sentences: int = 0,
        tokenizer: Tokenizer | None = None,
        *,
        segmenter: Segmenter = split_sentences,
        breadcrumbs: bool = True,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=breadcrumbs)
        self._validate_sizes(max_tokens)
        if sentences_per_chunk is not None and sentences_per_chunk <= 0:
            raise ValueError("sentences_per_chunk must be positive")
        if overlap_sentences < 0 or (
            sentences_per_chunk is not None and overlap_sentences >= sentences_per_chunk
        ):
            raise ValueError("overlap_sentences must be >= 0 and < sentences_per_chunk")
        self.max_tokens = max_tokens
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.segmenter = segmenter

    def _groups(self, index: DocIndex, sents: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        text = index.text
        splitter = RecursiveSplitter(self.max_tokens, tokenizer=self.tokenizer)
        groups: list[tuple[int, int]] = []
        i = 0
        while i < len(sents):
            j = i
            while j < len(sents):
                if self.sentences_per_chunk is not None and j - i >= self.sentences_per_chunk:
                    break
                if j > i and self.count(text[sents[i][0] : sents[j][1]]) > self.max_tokens:
                    break
                j += 1
            s, e = sents[i][0], sents[j - 1][1]
            if j - i == 1 and self.count(text[s:e]) > self.max_tokens and not self._is_atomic(index, s):
                groups.extend(splitter.split(text, s, e))
            else:
                groups.append((s, e))
            if j >= len(sents):
                break
            i = max(i + 1, j - self.overlap_sentences)
        return groups

    @staticmethod
    def _is_atomic(index: DocIndex, offset: int) -> bool:
        return index.span_at(offset).atomic

    def chunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        sents = _block_sentences(index, self.segmenter)
        return [
            self.make_chunk(index, position=i, start=s, end=e)
            for i, (s, e) in enumerate(self._groups(index, sents))
        ]


class ParagraphChunker(BaseChunker):
    """Paragraph = block (or blank-line run); merge short ones forward, split long ones."""

    name = "paragraph"

    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 50,
        tokenizer: Tokenizer | None = None,
        *,
        breadcrumbs: bool = True,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=breadcrumbs)
        self._validate_sizes(max_tokens)
        if not 0 <= min_tokens < max_tokens:
            raise ValueError("min_tokens must be in [0, max_tokens)")
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def chunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        text = index.text
        paras: list[tuple[int, int, bool]] = []
        for span in index.spans:
            if span.atomic:
                paras.append((span.start, span.end, True))
                continue
            pos = span.start
            for m in re.finditer(r"\n\s*\n", text[span.start : span.end]):
                paras.append((*trim_span(text, pos, span.start + m.start()), False))
                pos = span.start + m.end()
            paras.append((*trim_span(text, pos, span.end), False))
        splitter = RecursiveSplitter(self.max_tokens, tokenizer=self.tokenizer)
        ranges: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = None
        for s, e, atomic in paras:
            if s >= e:
                continue
            if cur is not None:
                if self.count(text[cur[0] : cur[1]]) < self.min_tokens and self.count(
                    text[cur[0] : e]
                ) <= self.max_tokens:
                    cur = (cur[0], e)
                    continue
                ranges.append(cur)
                cur = None
            if not atomic and self.count(text[s:e]) > self.max_tokens:
                ranges.extend(splitter.split(text, s, e))
            else:
                cur = (s, e)
        if cur is not None:
            ranges.append(cur)
        return [
            self.make_chunk(index, position=i, start=s, end=e) for i, (s, e) in enumerate(ranges)
        ]


class SlidingWindowChunker(BaseChunker):
    """Token windows with a large overlap (default 256 tokens, 50%); index grows 1/(1-overlap)."""

    name = "sliding_window"

    def __init__(
        self,
        window_tokens: int = 256,
        overlap: float = 0.5,
        tokenizer: Tokenizer | None = None,
        *,
        breadcrumbs: bool = True,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=breadcrumbs)
        self._validate_sizes(window_tokens)
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        self.window_tokens = window_tokens
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        spans = self.tokenizer.spans(index.text)
        step = max(1, int(self.window_tokens * (1 - self.overlap)))
        out: list[Chunk] = []
        i = 0
        while i < len(spans):
            j = min(i + self.window_tokens, len(spans))
            out.append(
                self.make_chunk(index, position=len(out), start=spans[i][0], end=spans[j - 1][1])
            )
            if j == len(spans):
                break
            i += step
        return out


class PageChunker(BaseChunker):
    """One page = one chunk (docs/02 §2.7); optionally recursive-split pages over ``max_tokens``.

    Documents without page information are treated as a single page.
    """

    name = "page"

    def __init__(
        self,
        max_tokens: int | None = None,
        tokenizer: Tokenizer | None = None,
        *,
        breadcrumbs: bool = True,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=breadcrumbs)
        if max_tokens is not None:
            self._validate_sizes(max_tokens)
        self.max_tokens = max_tokens

    def chunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        pages: list[tuple[int | None, int, int]] = []
        for span in index.spans:
            page = index.page_at(span.start)
            if pages and pages[-1][0] == page:
                pages[-1] = (page, pages[-1][1], span.end)
            else:
                pages.append((page, span.start, span.end))
        splitter = (
            RecursiveSplitter(self.max_tokens, tokenizer=self.tokenizer) if self.max_tokens else None
        )
        out: list[Chunk] = []
        for page, s, e in pages:
            ranges = splitter.split(index.text, s, e) if splitter else [trim_span(index.text, s, e)]
            for rs, re_ in ranges:
                out.append(
                    self.make_chunk(index, position=len(out), start=rs, end=re_, page=page)
                )
        return out
