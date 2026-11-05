"""Fixed-size baselines (docs/02-chunking.md §2.1–§2.2).

These ignore meaning and cut words / tables; they exist as the *floor* in
chunking experiments (and token splitting as a fallback inside recursion).
"""

from __future__ import annotations

from chatbot.chunking.base import BaseChunker, DocIndex, trim_span
from chatbot.core.tokens import Tokenizer
from chatbot.core.types import Chunk, Document


class FixedCharChunker(BaseChunker):
    """``text[i : i + N]`` with stride ``N - M`` (defaults: N=1000 chars, 15% overlap)."""

    name = "fixed_char"

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 150,
        tokenizer: Tokenizer | None = None,
        *,
        breadcrumbs: bool = False,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=breadcrumbs)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        text = index.text
        out: list[Chunk] = []
        step = self.chunk_size - self.overlap
        i = 0
        while i < len(text):
            s, e = trim_span(text, i, min(i + self.chunk_size, len(text)))
            if s < e:
                out.append(self.make_chunk(index, position=len(out), start=s, end=e))
            if i + self.chunk_size >= len(text):
                break
            i += step
        return out


class FixedTokenChunker(BaseChunker):
    """Windows of ``max_tokens`` tokens of the embedding tokenizer (defaults 512, ~12% overlap)."""

    name = "fixed_token"

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        tokenizer: Tokenizer | None = None,
        *,
        breadcrumbs: bool = False,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=breadcrumbs)
        self._validate_sizes(max_tokens, overlap_tokens)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        spans = self.tokenizer.spans(index.text)
        out: list[Chunk] = []
        step = self.max_tokens - self.overlap_tokens
        i = 0
        while i < len(spans):
            j = min(i + self.max_tokens, len(spans))
            out.append(
                self.make_chunk(index, position=len(out), start=spans[i][0], end=spans[j - 1][1])
            )
            if j == len(spans):
                break
            i += step
        return out
