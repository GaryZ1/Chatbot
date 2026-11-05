"""Context assembly under a token budget (docs/10-generation-context.md §1-§2).

Takes reranked chunks and produces the ``<sources>`` block: deduplicated, ordered
(best evidence first, grouped by document, or "edges" to counter lost-in-the-middle),
rendered with per-source metadata (title, section, updated, url) so the model can
reason about recency and authority, and **escaped** so chunk text can never close the
``<source>`` tag or inject markup. The resulting :class:`ContextPack` maps each
citation number ``n`` to its :class:`~chatbot.core.types.ScoredChunk`.
"""

from __future__ import annotations

import html
import logging
from collections.abc import Sequence
from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from chatbot.core.tokens import Tokenizer, default_tokenizer, truncate_to_tokens
from chatbot.core.types import ScoredChunk, sha256_text

logger = logging.getLogger(__name__)


class OrderStrategy(StrEnum):
    RELEVANCE = "relevance"  # best evidence first
    DOC_GROUPED = "doc_grouped"  # documents by best chunk score; chunks in document order
    EDGES = "edges"  # best at start and end, weakest in the middle


class ContextPack(BaseModel):
    xml: str
    #: Citation number → chunk, in rendering order (1-based).
    sources: dict[int, ScoredChunk] = Field(default_factory=dict)
    token_count: int = 0
    budget_tokens: int = 0
    #: Chunks left out because of the budget or ``max_sources``.
    dropped_ids: list[str] = Field(default_factory=list)
    #: Chunks removed as duplicates (same id or same normalised text).
    duplicate_ids: list[str] = Field(default_factory=list)
    truncated_ids: list[str] = Field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return [sc.chunk.chunk_id for sc in self.sources.values()]

    @property
    def is_empty(self) -> bool:
        return not self.sources


def _attr(value: str) -> str:
    return html.escape(" ".join(value.split()), quote=True)


def _fmt_date(v: object) -> str:
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return str(v) if v not in (None, "") else ""


def source_attributes(sc: ScoredChunk) -> dict[str, str]:
    c = sc.chunk
    md = c.metadata
    section = " > ".join(c.heading_path) or str(md.get("section") or "")
    updated = _fmt_date(md.get("updated_at") or md.get("updated") or md.get("date"))
    attrs = {
        "title": c.title or str(md.get("title") or ""),
        "section": section,
        "updated": updated,
        "url": c.source_uri or str(md.get("url") or ""),
    }
    if c.page is not None:
        attrs["page"] = str(c.page)
    return attrs


def render_source(n: int, sc: ScoredChunk, text: str | None = None) -> str:
    attrs = source_attributes(sc)
    attr_str = " ".join(f'{k}="{_attr(v)}"' for k, v in attrs.items() if v)
    body = html.escape((text if text is not None else sc.chunk.text).strip(), quote=False)
    head = f'<source id="{n}"' + (f" {attr_str}" if attr_str else "") + ">"
    return f"{head}\n{body}\n</source>"


class ContextBuilder:
    """Builds a :class:`ContextPack` within ``max_tokens``.

    Args:
        max_tokens: budget for the whole ``<sources>`` block (doc: a bounded, generous
            reranked set, e.g. 5-30k tokens).
        max_sources: hard cap on the number of sources.
        order: :class:`OrderStrategy`.
        min_chunk_tokens: a chunk that does not fit is truncated only if at least this
            many tokens of it fit; otherwise it is dropped.
        min_score: drop chunks scoring below this (``None`` disables).
    """

    def __init__(
        self,
        *,
        max_tokens: int = 8000,
        max_sources: int = 20,
        order: OrderStrategy = OrderStrategy.RELEVANCE,
        min_chunk_tokens: int = 64,
        min_score: float | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if max_tokens <= 0 or max_sources <= 0:
            raise ValueError("max_tokens and max_sources must be positive")
        self.max_tokens = max_tokens
        self.max_sources = max_sources
        self.order = order
        self.min_chunk_tokens = min_chunk_tokens
        self.min_score = min_score
        self.tokenizer = tokenizer or default_tokenizer()

    def _dedupe(self, chunks: Sequence[ScoredChunk]) -> tuple[list[ScoredChunk], list[str]]:
        seen_ids: set[str] = set()
        seen_hash: set[str] = set()
        out: list[ScoredChunk] = []
        dups: list[str] = []
        for sc in sorted(chunks, key=lambda s: s.score, reverse=True):
            h = sha256_text(sc.chunk.text)
            if sc.chunk.chunk_id in seen_ids or h in seen_hash:
                dups.append(sc.chunk.chunk_id)
                continue
            seen_ids.add(sc.chunk.chunk_id)
            seen_hash.add(h)
            out.append(sc)
        return out, dups

    def _order(self, selected: list[ScoredChunk]) -> list[ScoredChunk]:
        """``selected`` is in descending score order."""
        if self.order is OrderStrategy.DOC_GROUPED:
            groups: dict[str, list[ScoredChunk]] = {}
            for sc in selected:  # dict keeps first-seen (= best-scoring) doc order
                groups.setdefault(sc.chunk.doc_id, []).append(sc)
            return [
                sc
                for group in groups.values()
                for sc in sorted(group, key=lambda s: (s.chunk.position, s.chunk.char_start or 0))
            ]
        if self.order is OrderStrategy.EDGES:
            front: list[ScoredChunk] = []
            back: list[ScoredChunk] = []
            for i, sc in enumerate(selected):
                (front if i % 2 == 0 else back).append(sc)
            return front + back[::-1]
        return selected

    def build(self, chunks: Sequence[ScoredChunk]) -> ContextPack:
        candidates, dups = self._dedupe(chunks)
        dropped: list[str] = []
        if self.min_score is not None:
            dropped += [sc.chunk.chunk_id for sc in candidates if sc.score < self.min_score]
            candidates = [sc for sc in candidates if sc.score >= self.min_score]

        wrapper = self.tokenizer.count("<sources>\n</sources>")
        used = wrapper
        selected: list[tuple[ScoredChunk, str]] = []
        truncated: list[str] = []
        for sc in candidates:
            if len(selected) >= self.max_sources:
                dropped.append(sc.chunk.chunk_id)
                continue
            # Number placeholder: ids have at most 2-3 digits; count with a worst case.
            overhead = self.tokenizer.count(render_source(999, sc, text=""))
            text = sc.chunk.text.strip()
            cost = overhead + self.tokenizer.count(html.escape(text, quote=False))
            remaining = self.max_tokens - used
            if cost <= remaining:
                selected.append((sc, text))
                used += cost
                continue
            room = remaining - overhead
            if room >= self.min_chunk_tokens:
                cut = truncate_to_tokens(text, room, self.tokenizer)
                # Escaping can expand text; shrink until it fits.
                while cut and self.tokenizer.count(html.escape(cut, quote=False)) > room:
                    cut = truncate_to_tokens(
                        cut, max(1, self.tokenizer.count(cut) - 8), self.tokenizer
                    )
                if cut:
                    selected.append((sc, cut + " …"))
                    used += overhead + self.tokenizer.count(html.escape(cut + " …", quote=False))
                    truncated.append(sc.chunk.chunk_id)
                    continue
            dropped.append(sc.chunk.chunk_id)

        text_by_id = {sc.chunk.chunk_id: t for sc, t in selected}
        ordered = self._order([sc for sc, _ in selected])
        sources: dict[int, ScoredChunk] = {}
        rendered: list[str] = []
        for n, sc in enumerate(ordered, start=1):
            sources[n] = sc
            rendered.append(render_source(n, sc, text_by_id[sc.chunk.chunk_id]))
        xml = "<sources>\n" + "\n".join(rendered) + ("\n" if rendered else "") + "</sources>"
        token_count = self.tokenizer.count(xml)
        if dropped:
            logger.debug("context budget dropped %d chunks", len(dropped))
        return ContextPack(
            xml=xml,
            sources=sources,
            token_count=token_count,
            budget_tokens=self.max_tokens,
            dropped_ids=dropped,
            duplicate_ids=dups,
            truncated_ids=truncated,
        )
