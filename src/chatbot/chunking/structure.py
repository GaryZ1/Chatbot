"""Document-structure-aware chunking – the production default (docs/02-chunking.md §2.6, §4).

1. Headings delimit sections; each chunk carries its ``heading_path``.
2. Blocks of a section are packed up to ``max_tokens``; oversize prose blocks are
   recursively sub-split (:class:`RecursiveSplitter`).
3. Tables and code blocks are **never split** (oversize tables may optionally be
   split into row groups with the header repeated, docs/02 §2.18).
4. Pieces under ``min_tokens`` (~50) are merged with a neighbour – first within
   their section, then across sections (e.g. a heading directly followed by a
   sub-heading).
5. ``embed_text = "Title > H1 > H2\\n\\n" + [table summary] + text``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

from chatbot.chunking.base import (
    BaseChunker,
    DocIndex,
    Piece,
    merge_heading_paths,
)
from chatbot.chunking.recursive import DEFAULT_SEPARATORS, RecursiveSplitter
from chatbot.chunking.table import TableChunker, describe_table
from chatbot.core.tokens import Tokenizer
from chatbot.core.types import BlockType, Chunk, Document

logger = logging.getLogger(__name__)

TableStrategy = Literal["atomic", "row_groups"]


def _nested(a: Sequence[str], b: Sequence[str]) -> bool:
    ta, tb = tuple(a), tuple(b)
    return tb[: len(ta)] == ta or ta[: len(tb)] == tb


class StructureChunker(BaseChunker):
    """Heading-aware chunker with recursive sub-splitting and tiny-piece merging.

    Args:
        max_tokens: chunk budget (docs/02 §4 recommends 400–512).
        min_tokens: pieces below this are merged with a neighbour (~50).
        overlap_tokens: overlap used only when sub-splitting oversize prose blocks.
        table_strategy: ``"atomic"`` keeps every table whole (flagged ``oversize`` if
            needed); ``"row_groups"`` splits oversize tables with the header repeated.
    """

    name = "structure"

    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 50,
        overlap_tokens: int = 0,
        tokenizer: Tokenizer | None = None,
        *,
        table_strategy: TableStrategy = "atomic",
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        breadcrumbs: bool = True,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=breadcrumbs)
        self._validate_sizes(max_tokens, overlap_tokens)
        if not 0 <= min_tokens < max_tokens:
            raise ValueError("min_tokens must be in [0, max_tokens)")
        if table_strategy not in ("atomic", "row_groups"):
            raise ValueError("table_strategy must be 'atomic' or 'row_groups'")
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.table_strategy: TableStrategy = table_strategy
        self.separators = tuple(separators)

    # ------------------------------------------------------------------ units

    def units(self, index: DocIndex) -> list[Piece]:
        """One piece per block, tagged with section id, heading path and atomicity."""
        units: list[Piece] = []
        section = 0
        for span in index.spans:
            if span.block.type == BlockType.HEADING:
                section += 1
            summaries = [describe_table(span.block)] if span.block.type == BlockType.TABLE else []
            extra = {"block_type": span.block.type.value} if span.atomic else {}
            if span.block.type == BlockType.CODE and span.block.language:
                extra["code_language"] = span.block.language
            units.append(
                Piece(
                    start=span.start,
                    end=span.end,
                    heading_path=span.heading_path,
                    page=index.page_at(span.start),
                    atomic=span.atomic,
                    section=section,
                    tokens=self.count(span.block.text),
                    summaries=summaries,
                    extra=extra | {"_block": span.index},
                )
            )
        return units

    # ------------------------------------------------------------------ packing

    def pack(self, index: DocIndex, units: Sequence[Piece], max_tokens: int) -> list[Piece]:
        """Pack units into pieces ≤ ``max_tokens`` (atomic units excepted), then merge tiny ones."""
        text = index.text
        splitter = RecursiveSplitter(
            max_tokens, min(self.overlap_tokens, max_tokens - 1), self.separators, self.tokenizer
        )
        pieces: list[Piece] = []
        cur: Piece | None = None

        def flush() -> None:
            nonlocal cur
            if cur is not None:
                pieces.append(cur)
                cur = None

        for u in units:
            if u.text_override is not None:
                flush()
                pieces.append(u)
                continue
            if u.atomic:
                flush()
                pieces.extend(self._atomic_pieces(index, u, max_tokens))
                continue
            if u.tokens > max_tokens:
                flush()
                for s, e in splitter.split(text, u.start, u.end):
                    pieces.append(
                        Piece(
                            s,
                            e,
                            u.heading_path,
                            index.page_at(s),
                            section=u.section,
                            tokens=self.count(text[s:e]),
                        )
                    )
                continue
            if cur is not None and cur.section == u.section:
                tokens = self.count(text[cur.start : u.end])
                if tokens <= max_tokens:
                    cur.end = u.end
                    cur.tokens = tokens
                    continue
            flush()
            cur = Piece(u.start, u.end, u.heading_path, u.page, section=u.section, tokens=u.tokens)
        flush()
        pieces = self._merge_tiny(index, pieces, max_tokens, cross_section=False)
        return self._merge_tiny(index, pieces, max_tokens, cross_section=True)

    def _atomic_pieces(self, index: DocIndex, u: Piece, max_tokens: int) -> list[Piece]:
        block = index.doc.blocks[u.extra["_block"]]
        extra = {k: v for k, v in u.extra.items() if not k.startswith("_")}
        if u.tokens > max_tokens:
            if block.type == BlockType.TABLE and self.table_strategy == "row_groups":
                span = index.spans[index.span_index_at(u.start)]
                tc = TableChunker(max_tokens, tokenizer=self.tokenizer)
                groups = tc.pieces_for(span)
                for g in groups:
                    g.section = u.section
                    g.tokens = self.count(g.text_override or index.text[g.start : g.end])
                    g.extra = {**extra, **g.extra}
                return groups
            logger.debug("keeping oversize %s block atomic (%d tokens)", block.type, u.tokens)
            extra["oversize"] = True
        return [
            Piece(
                u.start,
                u.end,
                u.heading_path,
                u.page,
                atomic=True,
                section=u.section,
                tokens=u.tokens,
                summaries=list(u.summaries),
                extra=extra,
            )
        ]

    def _merge_tiny(
        self, index: DocIndex, pieces: list[Piece], max_tokens: int, *, cross_section: bool
    ) -> list[Piece]:
        if self.min_tokens <= 0 or len(pieces) < 2:
            return pieces
        text = index.text

        def can(a: Piece, b: Piece) -> bool:
            if a.text_override is not None or b.text_override is not None:
                return False
            if a.section != b.section:
                if not cross_section:
                    return False
                # never glue unrelated top-level sections together
                if not merge_heading_paths(a.heading_path, b.heading_path) and not (
                    _nested(a.heading_path, b.heading_path)
                ):
                    return False
            return self.count(text[a.start : b.end]) <= max_tokens

        def merge(a: Piece, b: Piece) -> Piece:
            return Piece(
                a.start,
                b.end,
                merge_heading_paths(
                    a.heading_path, b.heading_path, prefer_first=a.tokens >= b.tokens
                ),
                a.page if a.page is not None else b.page,
                atomic=a.atomic or b.atomic,
                section=b.section,
                tokens=self.count(text[a.start : b.end]),
                summaries=[*a.summaries, *b.summaries],
                extra={**a.extra, **b.extra},
            )

        out: list[Piece] = []
        carry: Piece | None = None

        def settle(p: Piece) -> None:  # try merging a tiny piece backwards
            if out and can(out[-1], p):
                out[-1] = merge(out[-1], p)
            else:
                out.append(p)

        for p in pieces:
            if carry is not None:
                if can(carry, p):
                    p = merge(carry, p)
                else:
                    settle(carry)
                carry = None
            if p.tokens < self.min_tokens:
                carry = p
            else:
                out.append(p)
        if carry is not None:
            settle(carry)
        return out

    # ------------------------------------------------------------------ public

    def pieces(self, index: DocIndex) -> list[Piece]:
        return self.pack(index, self.units(index), self.max_tokens)

    def chunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        if not index.spans:
            return []
        return self.chunks_from_pieces(index, self.pieces(index))
