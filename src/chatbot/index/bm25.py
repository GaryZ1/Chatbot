"""BM25F-lite inverted index (docs/07-retrieval.md §1, docs/05 §1 "Inverted index").

Two fields per chunk — ``title`` (title + heading path) and ``body``
(:attr:`Chunk.index_text`, i.e. contextualised embed text when present, which is
"contextual BM25") — combined BM25F-style::

    tf~(t, d)  = Σ_f  w_f · tf_f(t, d) / (1 − b + b · len_f(d) / avglen_f)
    score(q,d) = Σ_t  idf(t) · tf~ · (k1 + 1) / (k1 + tf~)
    idf(t)     = ln(1 + (N − df + 0.5) / (df + 0.5))

The same :class:`~chatbot.index.analyzer.Analyzer` tokenises documents and
queries. Visibility (tenant + ACL + metadata) is checked on every candidate
*before* top-K selection. Adds and deletes are incremental; ``save`` / ``load``
persist the chunks plus config (statistics are rebuilt on load).
"""

from __future__ import annotations

import asyncio
import heapq
import json
import logging
import math
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from chatbot.core.filters import Filter, visible
from chatbot.core.types import Chunk, Principal, ScoredChunk
from chatbot.index.analyzer import DEFAULT_ANALYZER, Analyzer

log = logging.getLogger(__name__)

_FORMAT_VERSION = 1


@dataclass
class _DocStats:
    title_tf: Counter[str]
    body_tf: Counter[str]
    title_len: int
    body_len: int


class BM25Index:
    """Implements :class:`~chatbot.core.interfaces.SparseIndex`; also a minimal ``DocStore``."""

    def __init__(
        self,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        title_boost: float = 2.0,
        title_b: float = 0.5,
        analyzer: Analyzer | None = None,
        name: str = "bm25",
    ) -> None:
        if k1 < 0 or not 0 <= b <= 1 or not 0 <= title_b <= 1 or title_boost < 0:
            raise ValueError("invalid BM25 parameters")
        self.k1 = k1
        self.b = b
        self.title_b = title_b
        self.title_boost = title_boost
        self.analyzer = analyzer or DEFAULT_ANALYZER
        self.name = name
        self._chunks: dict[str, Chunk] = {}
        self._stats: dict[str, _DocStats] = {}
        self._postings: dict[str, set[str]] = {}
        self._by_doc: dict[str, set[str]] = {}
        self._total_title = 0
        self._total_body = 0
        self._state = threading.RLock()
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------ analysis
    @staticmethod
    def title_text(chunk: Chunk) -> str:
        parts = [chunk.title, *chunk.heading_path]
        return " ".join(p for p in parts if p)

    def _index_one(self, chunk: Chunk) -> None:
        self._remove_one(chunk.chunk_id)
        title_terms = self.analyzer.analyze(self.title_text(chunk))
        body_terms = self.analyzer.analyze(chunk.index_text)
        st = _DocStats(Counter(title_terms), Counter(body_terms), len(title_terms), len(body_terms))
        self._chunks[chunk.chunk_id] = chunk
        self._stats[chunk.chunk_id] = st
        self._by_doc.setdefault(chunk.doc_id, set()).add(chunk.chunk_id)
        self._total_title += st.title_len
        self._total_body += st.body_len
        for term in set(st.title_tf) | set(st.body_tf):
            self._postings.setdefault(term, set()).add(chunk.chunk_id)

    def _remove_one(self, chunk_id: str) -> bool:
        chunk = self._chunks.pop(chunk_id, None)
        if chunk is None:
            return False
        st = self._stats.pop(chunk_id)
        self._total_title -= st.title_len
        self._total_body -= st.body_len
        for term in set(st.title_tf) | set(st.body_tf):
            plist = self._postings.get(term)
            if plist is not None:
                plist.discard(chunk_id)
                if not plist:
                    del self._postings[term]
        ids = self._by_doc.get(chunk.doc_id)
        if ids is not None:
            ids.discard(chunk_id)
            if not ids:
                del self._by_doc[chunk.doc_id]
        return True

    # ------------------------------------------------------------------ writes
    async def add(self, chunks: Sequence[Chunk]) -> None:
        async with self._write_lock:
            with self._state:
                for c in chunks:
                    self._index_one(c)

    async def delete_doc(self, doc_id: str) -> int:
        async with self._write_lock:
            with self._state:
                ids = list(self._by_doc.get(doc_id, ()))
                return sum(self._remove_one(cid) for cid in ids)

    async def delete(self, chunk_ids: Sequence[str]) -> int:
        async with self._write_lock:
            with self._state:
                return sum(self._remove_one(cid) for cid in chunk_ids)

    # ------------------------------------------------------------------ scoring
    def idf(self, term: str) -> float:
        n = len(self._chunks)
        df = len(self._postings.get(term, ()))
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _score(self, cid: str, qterms: Counter[str], idfs: dict[str, float]) -> float:
        st = self._stats[cid]
        n = max(len(self._chunks), 1)
        avg_t = max(self._total_title / n, 1e-9)
        avg_b = max(self._total_body / n, 1e-9)
        norm_t = 1 - self.title_b + self.title_b * st.title_len / avg_t
        norm_b = 1 - self.b + self.b * st.body_len / avg_b
        score = 0.0
        for term, qtf in qterms.items():
            tf = self.title_boost * st.title_tf.get(term, 0) / max(norm_t, 1e-9)
            tf += st.body_tf.get(term, 0) / max(norm_b, 1e-9)
            if tf > 0:
                score += qtf * idfs[term] * tf * (self.k1 + 1) / (self.k1 + tf)
        return score

    def score_texts(self, query: str) -> Counter[str]:
        """Analyzed query term counts (exposed for debugging / explainability)."""
        return Counter(self.analyzer.analyze(query))

    async def search(
        self, query: str, k: int, principal: Principal | None, flt: Filter | None = None
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []
        qterms = self.score_texts(query)
        if not qterms:
            return []
        with self._state:
            candidates: set[str] = set()
            for term in qterms:
                candidates |= self._postings.get(term, set())
            idfs = {t: self.idf(t) for t in qterms}
            # Pre-filter BEFORE scoring and top-K (ACL + tenant + metadata).
            allowed = [cid for cid in candidates if visible(self._chunks[cid], principal, flt)]
            scored = ((self._score(cid, qterms, idfs), cid) for cid in allowed)
            top = heapq.nlargest(k, scored, key=lambda t: (t[0], t[1]))
            return [
                ScoredChunk(
                    chunk=self._chunks[cid], score=s, source=self.name, stage_scores={self.name: s}
                )
                for s, cid in top
                if s > 0
            ]

    # ------------------------------------------------------------------ doc store
    async def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        with self._state:
            return [self._chunks[c] for c in chunk_ids if c in self._chunks]

    async def by_doc(self, doc_id: str) -> list[Chunk]:
        with self._state:
            chunks = [self._chunks[c] for c in self._by_doc.get(doc_id, ())]
        return sorted(chunks, key=lambda c: (c.kind.value, c.position, c.chunk_id))

    async def count(self) -> int:
        with self._state:
            return len(self._chunks)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    # ------------------------------------------------------------------ persistence
    def _config(self) -> dict[str, object]:
        return {
            "format": _FORMAT_VERSION,
            "k1": self.k1,
            "b": self.b,
            "title_b": self.title_b,
            "title_boost": self.title_boost,
            "analyzer": self.analyzer.signature,
            "name": self.name,
        }

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with self._state:
            lines = [c.model_dump_json() for c in self._chunks.values()]
        tmp = path / "bm25_chunks.jsonl.tmp"
        tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        tmp.replace(path / "bm25_chunks.jsonl")
        (path / "bm25.json").write_text(json.dumps(self._config(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, *, analyzer: Analyzer | None = None) -> BM25Index:
        path = Path(directory)
        cfg = json.loads((path / "bm25.json").read_text(encoding="utf-8"))
        if cfg.get("format") != _FORMAT_VERSION:
            raise ValueError(f"unsupported bm25 format {cfg.get('format')!r}")
        idx = cls(
            k1=float(cfg["k1"]),
            b=float(cfg["b"]),
            title_b=float(cfg["title_b"]),
            title_boost=float(cfg["title_boost"]),
            analyzer=analyzer,
            name=str(cfg.get("name", "bm25")),
        )
        if cfg.get("analyzer") != idx.analyzer.signature:
            raise ValueError(
                "analyzer mismatch between saved index and loader: indexing and querying "
                "must use the same analyzer"
            )
        with (path / "bm25_chunks.jsonl").open(encoding="utf-8") as fh:
            chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]
        idx._bulk(chunks)
        return idx

    def _bulk(self, chunks: Iterable[Chunk]) -> None:
        with self._state:
            for c in chunks:
                self._index_one(c)
