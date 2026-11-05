"""Small-to-big retrieval units (docs/02-chunking.md §2.12, §2.13, §4 step 5).

* :class:`ParentChildChunker` – structure-aware **parents** (≤ ~1,500 tokens,
  ``ChunkKind.PARENT``, returned to the LLM) and nested **children** (retrieval
  units, ``parent_id`` → parent). Children never cross parent boundaries and
  tables/code are never split at either level.
* :class:`SentenceWindowChunker` – index single sentences; the ±k sentence window
  is stored in ``metadata['window_text']`` (plus its offsets) for query-time expansion.
"""

from __future__ import annotations

from chatbot.chunking.base import BaseChunker, DocIndex, Piece
from chatbot.chunking.sentence import Segmenter, _block_sentences, split_sentences
from chatbot.chunking.structure import StructureChunker, TableStrategy
from chatbot.core.tokens import Tokenizer
from chatbot.core.types import Chunk, ChunkKind, Document


class ParentChildChunker(BaseChunker):
    """Parents for generation, children for retrieval.

    Output order: each parent followed by its children. Store both (DocStore);
    embed / BM25-index only ``kind == CHUNK``.
    """

    name = "parent_child"

    def __init__(
        self,
        child_max_tokens: int = 256,
        parent_max_tokens: int = 1500,
        min_tokens: int = 50,
        tokenizer: Tokenizer | None = None,
        *,
        child_min_tokens: int | None = None,
        table_strategy: TableStrategy = "atomic",
        breadcrumbs: bool = True,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=breadcrumbs)
        self._validate_sizes(child_max_tokens)
        if parent_max_tokens < child_max_tokens:
            raise ValueError("parent_max_tokens must be >= child_max_tokens")
        self.child_max_tokens = child_max_tokens
        self.parent_max_tokens = parent_max_tokens
        cmin = min(min_tokens, child_max_tokens // 4) if child_min_tokens is None else child_min_tokens
        self._parents = StructureChunker(
            parent_max_tokens,
            min(min_tokens, parent_max_tokens - 1),
            tokenizer=self.tokenizer,
            table_strategy=table_strategy,
            breadcrumbs=breadcrumbs,
        )
        self._children = StructureChunker(
            child_max_tokens,
            cmin,
            tokenizer=self.tokenizer,
            table_strategy=table_strategy,
            breadcrumbs=breadcrumbs,
        )
        self._parents.name = self._children.name = self.name

    @staticmethod
    def _clip(units: list[Piece], parent: Piece) -> list[Piece]:
        out: list[Piece] = []
        for u in units:
            if u.end <= parent.start or u.start >= parent.end:
                continue
            s, e = max(u.start, parent.start), min(u.end, parent.end)
            clipped = Piece(
                s,
                e,
                u.heading_path,
                u.page,
                atomic=u.atomic,
                section=u.section,
                summaries=list(u.summaries),
                extra=dict(u.extra),
            )
            out.append(clipped)
        return out

    def chunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        if not index.spans:
            return []
        units = self._children.units(index)
        parents = self._parents.pack(index, units, self.parent_max_tokens)
        out: list[Chunk] = []
        child_pos = 0
        for p_pos, parent in enumerate(parents):
            pchunk = self._parents.chunks_from_pieces(
                index, [parent], kind=ChunkKind.PARENT, start_position=p_pos
            )[0]
            if parent.text_override is not None:
                children = [parent]
            else:
                clipped = self._clip(units, parent)
                for c in clipped:
                    c.tokens = self.count(index.text[c.start : c.end])
                children = self._children.pack(index, clipped, self.child_max_tokens)
            cchunks = self._children.chunks_from_pieces(
                index, children, start_position=child_pos, parent_id=pchunk.chunk_id
            )
            pchunk.metadata["child_ids"] = [c.chunk_id for c in cchunks]
            child_pos += len(cchunks)
            out.append(pchunk)
            out.extend(cchunks)
        return out


class SentenceWindowChunker(BaseChunker):
    """One chunk per sentence with a ±``window`` sentence context for generation."""

    name = "sentence_window"

    def __init__(
        self,
        window: int = 2,
        tokenizer: Tokenizer | None = None,
        *,
        segmenter: Segmenter = split_sentences,
        breadcrumbs: bool = True,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=breadcrumbs)
        if window < 0:
            raise ValueError("window must be >= 0")
        self.window = window
        self.segmenter = segmenter

    def chunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        sents = _block_sentences(index, self.segmenter)
        out: list[Chunk] = []
        for i, (s, e) in enumerate(sents):
            ws = sents[max(0, i - self.window)][0]
            we = sents[min(len(sents) - 1, i + self.window)][1]
            out.append(
                self.make_chunk(
                    index,
                    position=i,
                    start=s,
                    end=e,
                    metadata={
                        "window_text": index.text[ws:we],
                        "window_start": ws,
                        "window_end": we,
                    },
                )
            )
        return out
