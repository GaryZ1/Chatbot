"""Late chunking (docs/02-chunking.md §2.16; Jina AI 2024, arXiv 2409.04701).

A long-context *token-level* embedding model encodes a whole window of the
document once; each chunk's vector is the mean of the token embeddings inside
its character span. Every chunk vector therefore "knows" its surrounding text
(anaphora such as "it" / "the company" are resolved) at zero LLM cost.

Boundaries come from a regular chunker (default :class:`StructureChunker`), so
``text``/offsets are unchanged — only the *vectors* differ. Use
:meth:`LateChunker.aembed` to get ``(chunks, vectors)`` and upsert those vectors
directly instead of re-embedding ``chunk.index_text``.

Long documents are processed in windows of whole chunks that fit the encoder's
``max_tokens`` (with ``context_chunks`` neighbours of extra context on each side).

**Fallback.** When no token-level model is available (most hosted APIs do not
expose token outputs) or the encoder fails, ``fallback_embedder`` embeds each
chunk's ``index_text`` (breadcrumb-prefixed, i.e. "naive" contextualisation);
chunks are marked ``metadata['late_chunking'] = False``. Late chunking helps
dense retrieval only – BM25 is unaffected.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict

from chatbot.chunking.base import BaseChunker, achunk_any, run_sync
from chatbot.chunking.structure import StructureChunker
from chatbot.core.interfaces import Chunker, Embedder
from chatbot.core.tokens import Tokenizer
from chatbot.core.types import Chunk, ChunkKind, Document

logger = logging.getLogger(__name__)


@runtime_checkable
class TokenEmbedder(Protocol):
    """Long-context encoder exposing per-token outputs (e.g. jina-embeddings-v3 self-hosted).

    ``embed_tokens`` returns ``(token_vectors[n_tokens, dim], offsets)`` where
    ``offsets[i] = (char_start, char_end)`` of token ``i`` within ``text``
    (special tokens excluded).
    """

    name: str
    dim: int
    max_tokens: int

    async def embed_tokens(self, text: str) -> tuple[np.ndarray, list[tuple[int, int]]]: ...


class LateChunkingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunks: list[Chunk]  # embedded chunks, aligned with ``vectors``
    vectors: np.ndarray  # shape (len(chunks), dim), L2-normalised
    #: chunks the base chunker produced that are stored but not embedded (e.g. parents)
    unembedded: list[Chunk] = []


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def pool_spans(
    token_vecs: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    spans: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Mean-pool the tokens overlapping each character span. Empty spans → zero vectors."""
    tv = np.asarray(token_vecs, dtype=np.float32)
    starts = np.array([o[0] for o in offsets], dtype=np.int64)
    ends = np.array([o[1] for o in offsets], dtype=np.int64)
    out = np.zeros((len(spans), tv.shape[1] if tv.ndim == 2 else 0), dtype=np.float32)
    for i, (s, e) in enumerate(spans):
        mask = (ends > s) & (starts < e)
        if mask.any():
            out[i] = tv[mask].mean(axis=0)
    return _normalise(out)


class LateChunker(BaseChunker):
    """Late chunking over boundaries from ``base_chunker``.

    Args:
        token_embedder: token-level long-context encoder (``None`` → fallback only).
        base_chunker: decides boundaries (default structure-aware, 512 tokens).
        fallback_embedder: regular embedder used when token outputs are unavailable.
        context_chunks: neighbouring chunks added as context on each side of a window.
        window_tokens: override the encoder window (defaults to ``token_embedder.max_tokens``),
            measured with this chunker's tokenizer with a 10% safety margin.
    """

    name = "late"

    def __init__(
        self,
        token_embedder: TokenEmbedder | None,
        base_chunker: Chunker | None = None,
        *,
        fallback_embedder: Embedder | None = None,
        context_chunks: int = 1,
        window_tokens: int | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=True)
        if token_embedder is None and fallback_embedder is None:
            raise ValueError("LateChunker needs a token_embedder or a fallback_embedder")
        if context_chunks < 0:
            raise ValueError("context_chunks must be >= 0")
        self.token_embedder = token_embedder
        self.fallback_embedder = fallback_embedder
        self.base_chunker: Chunker = base_chunker or StructureChunker(tokenizer=self.tokenizer)
        self.context_chunks = context_chunks
        limit = window_tokens or (token_embedder.max_tokens if token_embedder else 8192)
        if limit <= 0:
            raise ValueError("window_tokens must be positive")
        self.window_tokens = int(limit * 0.9)

    def chunk(self, doc: Document) -> list[Chunk]:
        result: list[Chunk] = run_sync(lambda: self.achunk(doc))
        return result

    async def achunk(self, doc: Document) -> list[Chunk]:
        res = await self.aembed(doc)
        return [*res.unembedded, *res.chunks]

    def _windows(self, chunks: Sequence[Chunk]) -> list[tuple[int, int]]:
        """Group consecutive chunks (by index) so each group fits the encoder window."""
        groups: list[tuple[int, int]] = []
        i = 0
        while i < len(chunks):
            j = i + 1
            total = chunks[i].token_count
            while j < len(chunks) and total + chunks[j].token_count <= self.window_tokens:
                total += chunks[j].token_count
                j += 1
            groups.append((i, j))
            i = j
        return groups

    async def _late_vectors(self, text: str, chunks: list[Chunk]) -> np.ndarray:
        assert self.token_embedder is not None
        vectors = np.zeros((len(chunks), self.token_embedder.dim), dtype=np.float32)
        for gi, gj in self._windows(chunks):
            ci = max(0, gi - self.context_chunks)
            cj = min(len(chunks), gj + self.context_chunks)
            ws = min(c.char_start or 0 for c in chunks[ci:cj])
            we = max(c.char_end or 0 for c in chunks[ci:cj])
            # shrink context if the window with neighbours exceeds the encoder limit
            if self.count(text[ws:we]) > self.window_tokens:
                ws = min(c.char_start or 0 for c in chunks[gi:gj])
                we = max(c.char_end or 0 for c in chunks[gi:gj])
            token_vecs, offsets = await self.token_embedder.embed_tokens(text[ws:we])
            spans = [((c.char_start or 0) - ws, (c.char_end or 0) - ws) for c in chunks[gi:gj]]
            vectors[gi:gj] = pool_spans(token_vecs, offsets, spans)
        return vectors

    async def aembed(self, doc: Document) -> LateChunkingResult:
        """Chunk ``doc`` and return late-chunked vectors aligned with ``chunks``."""
        chunks = await achunk_any(self.base_chunker, doc)
        targets = [
            c
            for c in chunks
            if c.kind == ChunkKind.CHUNK and c.char_start is not None and c.char_end is not None
        ]
        target_ids = {c.chunk_id for c in targets}
        unembedded = [c for c in chunks if c.chunk_id not in target_ids]
        if not targets:
            return LateChunkingResult(
                chunks=[], vectors=np.zeros((0, 0), dtype=np.float32), unembedded=unembedded
            )
        text = doc.text
        vectors: np.ndarray | None = None
        late = False
        if self.token_embedder is not None:
            try:
                vectors = await self._late_vectors(text, targets)
                late = True
            except (NotImplementedError, RuntimeError, ValueError) as exc:
                if self.fallback_embedder is None:
                    raise
                logger.warning("late chunking failed for %s (%s); using fallback", doc.doc_id, exc)
        if vectors is None:
            assert self.fallback_embedder is not None
            vectors = _normalise(
                np.asarray(
                    await self.fallback_embedder.embed_documents([c.index_text for c in targets]),
                    dtype=np.float32,
                )
            )
            late = False
        model = self.token_embedder.name if late and self.token_embedder else (
            self.fallback_embedder.name if self.fallback_embedder else ""
        )
        for c in targets:
            c.metadata["late_chunking"] = late
            c.metadata["embedding_model"] = model
        return LateChunkingResult(chunks=targets, vectors=vectors, unembedded=unembedded)
