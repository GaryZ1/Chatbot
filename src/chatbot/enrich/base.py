"""Enrichment step protocol and the versioned :class:`Enricher` pipeline (docs/03-enrichment.md §3).

Rules implemented here:

1. *Separate embedding text from display text* – steps only write ``embed_text``
   and ``metadata``; ``Chunk.text`` (what is shown and cited) is never modified.
2. *Version the enrichment* – the pipeline version is derived from the ordered
   step names + versions and stamped on every chunk (``enrichment_version``), so a
   change in any step forces a re-run / re-index.
5. Each step must earn its place on the golden set; steps are composable so they
   can be ablated individually.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from chatbot.chunking.base import breadcrumb
from chatbot.core.types import Chunk, ChunkKind, Document

logger = logging.getLogger(__name__)


@runtime_checkable
class EnrichmentStep(Protocol):
    """One enrichment. ``apply`` returns the full chunk list (it may append new chunks)."""

    name: str
    version: str

    async def apply(self, doc: Document, chunks: list[Chunk]) -> list[Chunk]: ...


def chunk_breadcrumb(chunk: Chunk) -> str:
    """``"Title > H1 > H2"`` for a chunk."""
    return breadcrumb(chunk.title, chunk.heading_path)


def compose_embed_text(chunk: Chunk, *parts: str | None) -> str:
    """``breadcrumb + parts + text`` – the canonical embedding text layout (docs/03 §3.1)."""
    pieces = [chunk_breadcrumb(chunk), *(p for p in parts if p), chunk.text]
    return "\n\n".join(p for p in pieces if p)


class BreadcrumbStep:
    """Zero-cost header injection: ``embed_text = breadcrumb + text`` when not already set."""

    name = "breadcrumb"
    version = "1"

    def __init__(
        self, kinds: Sequence[ChunkKind] = (ChunkKind.CHUNK,), *, overwrite: bool = False
    ) -> None:
        self.kinds = frozenset(kinds)
        self.overwrite = overwrite

    async def apply(self, doc: Document, chunks: list[Chunk]) -> list[Chunk]:
        for c in chunks:
            if c.kind in self.kinds and (self.overwrite or not c.embed_text):
                c.embed_text = compose_embed_text(c)
        return chunks


class Enricher:
    """Runs steps in order over a document's chunks and stamps the pipeline version.

    Chunks are deep-copied first, so callers' objects are never mutated.
    """

    def __init__(self, steps: Sequence[EnrichmentStep], *, name: str = "enrich") -> None:
        names = [s.name for s in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate enrichment step names: {names}")
        self.steps = list(steps)
        self.name = name

    @property
    def version(self) -> str:
        """Stable id of the ordered step configuration, e.g. ``enrich-3f2a…``."""
        spec = "|".join(f"{s.name}@{s.version}" for s in self.steps)
        return f"{self.name}-" + hashlib.sha256(spec.encode()).hexdigest()[:12]

    @property
    def spec(self) -> list[str]:
        return [f"{s.name}@{s.version}" for s in self.steps]

    async def run(self, doc: Document, chunks: Sequence[Chunk]) -> list[Chunk]:
        out = [c.model_copy(deep=True) for c in chunks]
        for step in self.steps:
            before = len(out)
            out = await step.apply(doc, out)
            logger.debug("enrich step %s: %d → %d chunks", step.name, before, len(out))
        version = self.version
        for c in out:
            c.enrichment_version = version
            c.metadata["enrichment_version"] = version
            c.metadata["enrichment_steps"] = self.spec
        return out
