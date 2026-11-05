"""Shared chunking machinery (docs/02-chunking.md §4, step 6).

Every chunker produces :class:`~chatbot.core.types.Chunk` objects whose
``char_start``/``char_end`` index into ``Document.text`` (blocks joined with
``"\\n\\n"``), so that for contiguous chunkers ``doc.text[c.char_start:c.char_end]
== c.text`` exactly — the property citations and highlighting depend on.

:class:`DocIndex` maps character offsets back to blocks, heading paths and pages.
:class:`BaseChunker` builds chunks with deterministic ids (:meth:`Chunk.make_id`)
and copies ACL, tenant, provenance and filterable metadata (``doc_type``,
``language``, ``updated_at``, ``is_latest``) from the document.
"""

from __future__ import annotations

import abc
import asyncio
import bisect
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from chatbot.core.tokens import Tokenizer, default_tokenizer
from chatbot.core.types import Block, BlockType, Chunk, ChunkKind, Document

logger = logging.getLogger(__name__)

#: Block types that must never be split (docs/02 §6 anti-patterns).
ATOMIC_TYPES: frozenset[BlockType] = frozenset({BlockType.TABLE, BlockType.CODE})
BLOCK_SEPARATOR = "\n\n"
BREADCRUMB_SEPARATOR = " > "


@dataclass(frozen=True, slots=True)
class BlockSpan:
    """A block's location inside ``Document.text``."""

    index: int  # index into doc.blocks
    block: Block
    start: int
    end: int
    heading_path: tuple[str, ...]
    page: int | None

    @property
    def atomic(self) -> bool:
        return self.block.type in ATOMIC_TYPES


def block_spans(doc: Document) -> list[BlockSpan]:
    """Offsets of every text-bearing block in ``doc.text`` plus its heading path.

    Mirrors :attr:`Document.text` exactly (blocks with empty text are skipped and
    the rest are joined by ``"\\n\\n"``). A heading's own path includes itself.
    """
    spans: list[BlockSpan] = []
    stack: list[tuple[int, str]] = []
    pos = 0
    for i, b in enumerate(doc.blocks):
        if b.type == BlockType.HEADING and b.text.strip():
            level = b.level or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, " ".join(b.text.split())))
        if not b.text:
            continue
        if spans:
            pos += len(BLOCK_SEPARATOR)
        start, end = pos, pos + len(b.text)
        spans.append(BlockSpan(i, b, start, end, tuple(t for _, t in stack), b.page))
        pos = end
    return spans


class DocIndex:
    """Offset → block / heading path / page lookups over one document."""

    def __init__(self, doc: Document) -> None:
        self.doc = doc
        self.spans = block_spans(doc)
        self.text = BLOCK_SEPARATOR.join(s.block.text for s in self.spans)
        self._starts = [s.start for s in self.spans]
        # carry the last known page forward for blocks without page info
        pages: list[int | None] = []
        last: int | None = None
        for s in self.spans:
            last = s.page if s.page is not None else last
            pages.append(last)
        self._pages = pages

    def span_index_at(self, offset: int) -> int:
        if not self.spans:
            raise ValueError("document has no text blocks")
        return max(0, bisect.bisect_right(self._starts, offset) - 1)

    def span_at(self, offset: int) -> BlockSpan:
        return self.spans[self.span_index_at(offset)]

    def heading_path_at(self, offset: int) -> tuple[str, ...]:
        return self.span_at(offset).heading_path if self.spans else ()

    def page_at(self, offset: int) -> int | None:
        return self._pages[self.span_index_at(offset)] if self.spans else None

    def spans_in(self, start: int, end: int) -> list[BlockSpan]:
        """Blocks overlapping ``[start, end)``."""
        if not self.spans:
            return []
        i = self.span_index_at(start)
        out: list[BlockSpan] = []
        while i < len(self.spans) and self.spans[i].start < end:
            if self.spans[i].end > start:
                out.append(self.spans[i])
            i += 1
        return out


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink ``[start, end)`` so it neither starts nor ends with whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def breadcrumb(title: str, heading_path: Sequence[str]) -> str:
    """``"Title > H1 > H2"`` (the title is not repeated when it equals the first heading)."""
    parts = [" ".join(title.split())] if title.strip() else []
    for h in heading_path:
        h = " ".join(h.split())
        if h and (not parts or parts[-1].casefold() != h.casefold()):
            parts.append(h)
    return BREADCRUMB_SEPARATOR.join(parts)


def merge_heading_paths(
    a: Sequence[str], b: Sequence[str], *, prefer_first: bool = False
) -> tuple[str, ...]:
    """Heading path for a chunk spanning two pieces.

    Nested paths (one is a prefix of the other) resolve to the path of the dominant
    piece: ``prefer_first`` keeps ``a`` (e.g. a tiny sub-section merged backwards into
    its parent's prose), otherwise the deeper path wins (a heading-only piece merged
    forwards into its sub-section). Unrelated paths resolve to their common prefix.
    """
    ta, tb = tuple(a), tuple(b)
    if tb[: len(ta)] == ta or ta[: len(tb)] == tb:
        if prefer_first:
            return ta
        return tb if len(tb) >= len(ta) else ta
    common: list[str] = []
    for x, y in zip(ta, tb, strict=False):
        if x != y:
            break
        common.append(x)
    return tuple(common)


def doc_metadata(doc: Document) -> dict[str, Any]:
    """Filterable metadata every chunk inherits from its document (docs/03 §1, docs/05 §3)."""
    meta: dict[str, Any] = dict(doc.metadata)
    meta.update(
        {
            "doc_type": doc.doc_type,
            "language": doc.language,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "author": doc.author,
            "is_latest": True,
            "content_hash": doc.content_hash,
            "parse_quality": doc.parse_quality.score,
            "breadcrumbs": list(doc.breadcrumbs),
        }
    )
    return meta


@runtime_checkable
class AsyncChunker(Protocol):
    """Chunkers that need I/O (embedders, LLMs) implement ``achunk``; all chunkers here do."""

    name: str

    async def achunk(self, doc: Document) -> list[Chunk]: ...


@dataclass(slots=True)
class Piece:
    """An intermediate chunk: a span of ``DocIndex.text`` with structural context."""

    start: int
    end: int
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    atomic: bool = False
    section: int = 0
    tokens: int = 0
    text_override: str | None = None  # e.g. table row group with repeated header
    summaries: list[str] = field(default_factory=list)  # table summaries for embed_text
    extra: dict[str, Any] = field(default_factory=dict)


class BaseChunker(abc.ABC):
    """Base class: tokenizer, id/metadata plumbing and an async adapter.

    Args:
        tokenizer: the *embedding model's* tokenizer (docs/02 §2.2).
        breadcrumbs: prefix ``embed_text`` with ``"Title > H1 > H2"``.
    """

    name: str = "base"

    def __init__(self, tokenizer: Tokenizer | None = None, *, breadcrumbs: bool = True) -> None:
        self.tokenizer = tokenizer or default_tokenizer()
        self.breadcrumbs = breadcrumbs

    @abc.abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` into chunks (synchronous, CPU-only chunkers)."""

    async def achunk(self, doc: Document) -> list[Chunk]:
        """Async entry point used by the pipeline; CPU chunkers run inline."""
        return self.chunk(doc)

    def count(self, text: str) -> int:
        return self.tokenizer.count(text)

    @staticmethod
    def _validate_sizes(max_tokens: int, overlap: int = 0) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap < 0 or overlap >= max_tokens:
            raise ValueError("overlap must be in [0, max_tokens)")

    def build_embed_text(
        self, doc: Document, heading_path: Sequence[str], text: str, extras: Sequence[str] = ()
    ) -> str | None:
        """``breadcrumb + extras + text``; ``None`` when there is nothing to add."""
        parts: list[str] = []
        if self.breadcrumbs:
            crumb = breadcrumb(doc.title, heading_path)
            if crumb:
                parts.append(crumb)
        parts.extend(e for e in extras if e)
        if not parts:
            return None
        return "\n\n".join([*parts, text])

    def make_chunk(
        self,
        index: DocIndex,
        *,
        position: int,
        start: int,
        end: int,
        text: str | None = None,
        kind: ChunkKind = ChunkKind.CHUNK,
        heading_path: Sequence[str] | None = None,
        page: int | None = None,
        parent_id: str | None = None,
        embed_text: str | None = None,
        embed_extras: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
        salt: str = "",
    ) -> Chunk:
        """Create a chunk for ``index.text[start:end]`` with inherited document fields."""
        doc = index.doc
        if not 0 <= start <= end <= len(index.text):
            raise ValueError(f"invalid span [{start}, {end}) for text of length {len(index.text)}")
        if text is None:  # never start/end a chunk on whitespace; keeps offsets exact
            start, end = trim_span(index.text, start, end)
        body = index.text[start:end] if text is None else text
        path = tuple(heading_path) if heading_path is not None else index.heading_path_at(start)
        pg = page if page is not None else index.page_at(start)
        meta = doc_metadata(doc)
        meta["chunker"] = self.name
        if metadata:
            meta.update(metadata)
        if embed_text is None:
            embed_text = self.build_embed_text(doc, path, body, embed_extras)
        return Chunk(
            chunk_id=Chunk.make_id(
                doc.doc_id, doc.version, position, kind.value, f"{self.name}:{start}:{end}:{salt}"
            ),
            doc_id=doc.doc_id,
            doc_version=doc.version,
            kind=kind,
            parent_id=parent_id,
            position=position,
            text=body,
            embed_text=embed_text,
            title=doc.title,
            heading_path=list(path),
            page=pg,
            char_start=start,
            char_end=end,
            token_count=self.count(body),
            tenant_id=doc.tenant_id,
            acl=doc.acl.model_copy(deep=True),
            source_uri=doc.source_uri,
            metadata=meta,
        )

    def chunks_from_pieces(
        self,
        index: DocIndex,
        pieces: Sequence[Piece],
        *,
        kind: ChunkKind = ChunkKind.CHUNK,
        start_position: int = 0,
        parent_id: str | None = None,
    ) -> list[Chunk]:
        out: list[Chunk] = []
        for i, p in enumerate(pieces):
            meta = dict(p.extra)
            if p.atomic:
                meta["atomic"] = True
            out.append(
                self.make_chunk(
                    index,
                    position=start_position + i,
                    start=p.start,
                    end=p.end,
                    text=p.text_override,
                    kind=kind,
                    heading_path=p.heading_path,
                    page=p.page,
                    parent_id=parent_id,
                    embed_extras=p.summaries,
                    metadata=meta,
                )
            )
        return out


def run_sync(coro_factory: Any) -> Any:
    """Run an async chunker from sync code; refuses inside a running loop (use ``achunk``)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    raise RuntimeError(
        "this chunker is async; call 'await chunker.achunk(doc)' inside an event loop"
    )


async def achunk_any(chunker: Any, doc: Document) -> list[Chunk]:
    """Chunk with ``achunk`` when the chunker has it, else the sync ``chunk``."""
    fn = getattr(chunker, "achunk", None)
    if fn is not None:
        return list(await fn(doc))
    return list(chunker.chunk(doc))
