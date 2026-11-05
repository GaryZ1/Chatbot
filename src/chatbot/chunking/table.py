"""Table-aware chunking (docs/02-chunking.md §2.18, docs/01 §3 "Tables").

Tables are atomic chunks. Tables over the token budget are split into **row
groups with the header row repeated**, and a natural-language summary plus a
column description goes into ``embed_text`` (never into the display text).

Row-group chunks are the one documented exception to exact offsets: their
``text`` is ``header + rows`` while ``char_start``/``char_end`` cover the rows
in the source (``metadata['header_repeated'] = True``).
"""

from __future__ import annotations

import re

from chatbot.chunking.base import BaseChunker, BlockSpan, DocIndex, Piece
from chatbot.core.tokens import Tokenizer
from chatbot.core.types import Block, BlockType, Chunk, Document

_SEP_LINE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")


def split_table_lines(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Return ``(header_lines, row_line_spans)`` for a pipe / line-per-row table.

    The header is the first line plus the ``|---|`` separator when present.
    Row spans are ``(start, end)`` offsets *within* ``text``.
    """
    lines: list[tuple[int, int]] = []
    pos = 0
    for line in text.split("\n"):
        lines.append((pos, pos + len(line)))
        pos += len(line) + 1
    lines = [(s, e) for s, e in lines if text[s:e].strip()]
    if not lines:
        return [], []
    n_header = 2 if len(lines) > 1 and _SEP_LINE.match(text[lines[1][0] : lines[1][1]]) else 1
    header = [text[s:e] for s, e in lines[:n_header]]
    return header, lines[n_header:]


def _cells(line: str) -> list[str]:
    s = line.strip().strip("|")
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)] if "|" in line else [line.strip()]


def describe_table(block: Block, *, max_columns: int = 12) -> str:
    """NL description for embedding: the block's own summary, else columns + row count."""
    if block.summary:
        return block.summary
    header, rows = split_table_lines(block.text)
    cols = [c for c in _cells(header[0]) if c] if header else []
    parts = []
    if block.caption:
        parts.append(f"Table: {block.caption}.")
    if cols:
        shown = ", ".join(cols[:max_columns]) + (" …" if len(cols) > max_columns else "")
        parts.append(f"Table with columns {shown}; {len(rows)} rows.")
    else:
        parts.append(f"Table with {len(rows)} rows.")
    return " ".join(parts)


def table_row_groups(
    span: BlockSpan, tokenizer: Tokenizer, max_tokens: int, *, rows_per_group: int | None = None
) -> list[Piece]:
    """Split a table block into row groups, each prefixed by the header lines."""
    text = span.block.text
    header, rows = split_table_lines(text)
    header_text = "\n".join(header)
    summary = describe_table(span.block)
    if not rows:
        return [
            Piece(span.start, span.end, span.heading_path, span.page, atomic=True, summaries=[summary])
        ]
    groups: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = []
    for r in rows:
        candidate = [*cur, r]
        body = header_text + "\n" + text[candidate[0][0] : candidate[-1][1]]
        too_many = rows_per_group is not None and len(candidate) > rows_per_group
        if cur and (too_many or tokenizer.count(body) > max_tokens):
            groups.append(cur)
            cur = [r]
        else:
            cur = candidate
    if cur:
        groups.append(cur)
    pieces: list[Piece] = []
    first_row = 0
    for g in groups:
        rs, re_ = g[0][0], g[-1][1]
        pieces.append(
            Piece(
                start=span.start + rs,
                end=span.start + re_,
                heading_path=span.heading_path,
                page=span.page,
                atomic=True,
                text_override=header_text + "\n" + text[rs:re_],
                summaries=[f"{summary} Rows {first_row + 1}-{first_row + len(g)}."],
                extra={
                    "header_repeated": True,
                    "row_start": first_row,
                    "row_end": first_row + len(g),
                    "table_block": span.index,
                },
            )
        )
        first_row += len(g)
    return pieces


class TableChunker(BaseChunker):
    """Emits chunks for the document's ``TABLE`` blocks only.

    Use it as a dedicated table index, or through
    :class:`~chatbot.chunking.structure.StructureChunker` (``table_strategy="row_groups"``).
    """

    name = "table"

    def __init__(
        self,
        max_tokens: int = 512,
        rows_per_group: int | None = None,
        tokenizer: Tokenizer | None = None,
        *,
        breadcrumbs: bool = True,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=breadcrumbs)
        self._validate_sizes(max_tokens)
        if rows_per_group is not None and rows_per_group <= 0:
            raise ValueError("rows_per_group must be positive")
        self.max_tokens = max_tokens
        self.rows_per_group = rows_per_group

    def pieces_for(self, span: BlockSpan) -> list[Piece]:
        if self.count(span.block.text) <= self.max_tokens and self.rows_per_group is None:
            return [
                Piece(
                    span.start,
                    span.end,
                    span.heading_path,
                    span.page,
                    atomic=True,
                    summaries=[describe_table(span.block)],
                    extra={"table_block": span.index},
                )
            ]
        return table_row_groups(
            span, self.tokenizer, self.max_tokens, rows_per_group=self.rows_per_group
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        pieces: list[Piece] = []
        for span in index.spans:
            if span.block.type == BlockType.TABLE:
                pieces.extend(self.pieces_for(span))
        chunks = self.chunks_from_pieces(index, pieces)
        for c, p in zip(chunks, pieces, strict=True):
            if p.text_override is None and index.doc.blocks[p.extra["table_block"]].html:
                c.metadata["table_html"] = index.doc.blocks[p.extra["table_block"]].html
        return chunks
