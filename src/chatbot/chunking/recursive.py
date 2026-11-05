"""Recursive separator splitting – the industry-default baseline (docs/02-chunking.md §2.5).

Separators are tried in priority order (``\\n\\n`` → ``\\n`` → sentence ends →
clauses → spaces → hard token split) and recursion only happens on pieces that
are still above the token limit. Splitting works on *offsets*, never on copies
of the text, so every chunk maps back to the source exactly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from chatbot.chunking.base import BaseChunker, DocIndex, trim_span
from chatbot.core.tokens import Tokenizer, default_tokenizer
from chatbot.core.types import Chunk, Document

logger = logging.getLogger(__name__)

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


class RecursiveSplitter:
    """Offset-preserving recursive splitter used by most chunkers in this package.

    Args:
        max_tokens: hard upper bound per piece (measured with ``tokenizer``).
        overlap_tokens: approximate token overlap between consecutive pieces.
        separators: priority-ordered separators; ``""`` means a hard token split.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.separators = tuple(separators) if separators else ("",)
        self.tokenizer = tokenizer or default_tokenizer()

    def _count(self, text: str, s: int, e: int) -> int:
        return self.tokenizer.count(text[s:e])

    def split(self, text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
        """Split ``text[start:end]`` into trimmed spans of at most ``max_tokens`` tokens."""
        end = len(text) if end is None else end
        s, e = trim_span(text, start, end)
        if s >= e:
            return []
        if self._count(text, s, e) <= self.max_tokens:
            return [(s, e)]
        return self._split(text, s, e, 0)

    def _split(self, text: str, s: int, e: int, sep_idx: int) -> list[tuple[int, int]]:
        idx = sep_idx
        while idx < len(self.separators):
            sep = self.separators[idx]
            if sep == "" or text.find(sep, s, e) != -1:
                break
            idx += 1
        if idx >= len(self.separators) or self.separators[idx] == "":
            return self.hard_split(text, s, e)
        sep = self.separators[idx]
        atoms: list[tuple[int, int]] = []
        for ps, pe in self._cut(text, s, e, sep):
            ps, pe = trim_span(text, ps, pe)
            if ps >= pe:
                continue
            if self._count(text, ps, pe) > self.max_tokens:
                atoms.extend(self._split(text, ps, pe, idx + 1))
            else:
                atoms.append((ps, pe))
        return self._merge(text, atoms)

    @staticmethod
    def _cut(text: str, s: int, e: int, sep: str) -> list[tuple[int, int]]:
        """Cut after every occurrence of ``sep`` (separator stays with the left piece)."""
        out: list[tuple[int, int]] = []
        pos = s
        while True:
            j = text.find(sep, pos, e)
            if j == -1:
                break
            out.append((pos, j + len(sep)))
            pos = j + len(sep)
        if pos < e:
            out.append((pos, e))
        return out

    def _merge(self, text: str, atoms: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Greedily pack atoms up to ``max_tokens`` with an optional token overlap."""
        out: list[tuple[int, int]] = []
        window: list[tuple[int, int]] = []
        for atom in atoms:
            if window and self._count(text, window[0][0], atom[1]) > self.max_tokens:
                out.append((window[0][0], window[-1][1]))
                if self.overlap_tokens > 0:
                    window = window[1:]
                    while window and self._count(text, window[0][0], window[-1][1]) > self.overlap_tokens:
                        window = window[1:]
                    while window and self._count(text, window[0][0], atom[1]) > self.max_tokens:
                        window = window[1:]
                else:
                    window = []
            window.append(atom)
        if window:
            out.append((window[0][0], window[-1][1]))
        return out

    def hard_split(self, text: str, s: int, e: int) -> list[tuple[int, int]]:
        """Split on tokenizer token boundaries (last resort; may cut sentences)."""
        spans = self.tokenizer.spans(text[s:e])
        if not spans:
            return [(s, e)]
        step = max(1, self.max_tokens - self.overlap_tokens)
        out: list[tuple[int, int]] = []
        i = 0
        while i < len(spans):
            j = min(i + self.max_tokens, len(spans))
            out.append((s + spans[i][0], s + spans[j - 1][1]))
            if j == len(spans):
                break
            i += step
        return out


class RecursiveChunker(BaseChunker):
    """Recursive splitting over the whole document text (structure-unaware baseline).

    Defaults follow docs/02 §2.5: 512 tokens, no overlap (Chroma found overlap adds little).
    """

    name = "recursive"

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        tokenizer: Tokenizer | None = None,
        *,
        breadcrumbs: bool = True,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=breadcrumbs)
        self.splitter = RecursiveSplitter(max_tokens, overlap_tokens, separators, self.tokenizer)

    def chunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        spans = self.splitter.split(index.text)
        return [
            self.make_chunk(index, position=i, start=s, end=e) for i, (s, e) in enumerate(spans)
        ]
