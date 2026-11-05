"""Canonical data model shared by every pipeline stage.

The document schema follows docs/01-ingestion-parsing.md §5 (typed blocks with
hierarchy, provenance, ACLs, content hash). Chunks follow docs/02-chunking.md §4
(ids, parent ids, positions, heading paths, pages and character offsets for
citations) and the index design in docs/05-indexing-vector-stores.md §3.
"""

from __future__ import annotations

import hashlib
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


def utcnow() -> datetime:
    return datetime.now(UTC)


def sha256_text(text: str) -> str:
    """SHA-256 over NFKC-normalised, whitespace-collapsed text (exact-dedup key)."""
    norm = " ".join(unicodedata.normalize("NFKC", text).split())
    return "sha256:" + hashlib.sha256(norm.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- identity / ACL


class Principal(BaseModel):
    """The authenticated caller. Group memberships are resolved by the IdP at request time."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    groups: frozenset[str] = frozenset()
    tenant_id: str = "default"
    roles: frozenset[str] = frozenset()

    def scope_key(self) -> str:
        """Stable key for caches that must never be shared across permission sets."""
        raw = "|".join([self.tenant_id, self.user_id, ",".join(sorted(self.groups))])
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


class ACL(BaseModel):
    """Per-document access control. Deny always wins over allow."""

    public: bool = False
    allow_users: list[str] = Field(default_factory=list)
    allow_groups: list[str] = Field(default_factory=list)
    deny_users: list[str] = Field(default_factory=list)
    deny_groups: list[str] = Field(default_factory=list)

    def permits(self, principal: Principal) -> bool:
        if principal.user_id in self.deny_users:
            return False
        if principal.groups.intersection(self.deny_groups):
            return False
        if self.public:
            return True
        if principal.user_id in self.allow_users:
            return True
        return bool(principal.groups.intersection(self.allow_groups))


# --------------------------------------------------------------------------- documents


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    FIGURE = "figure"
    PAGE_BREAK = "page_break"


class Block(BaseModel):
    type: BlockType
    text: str = ""
    level: int | None = None  # heading level (1..6)
    page: int | None = None
    html: str | None = None  # tables: HTML keeps merged cells
    summary: str | None = None  # tables/figures: NL description for embedding
    caption: str | None = None
    image_ref: str | None = None
    language: str | None = None  # code blocks
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseQuality(BaseModel):
    parser: str = "unknown"
    score: float = 1.0  # 0..1; < threshold routes to a VLM fallback
    flags: list[str] = Field(default_factory=list)


class Document(BaseModel):
    doc_id: str
    version: str = "1"
    source_uri: str = ""
    title: str = ""
    breadcrumbs: list[str] = Field(default_factory=list)
    language: str = "en"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    author: str | None = None
    tenant_id: str = "default"
    acl: ACL = Field(default_factory=ACL)
    doc_type: str = "other"
    blocks: list[Block] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parse_quality: ParseQuality = Field(default_factory=ParseQuality)
    deleted: bool = False  # tombstone from a connector

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        return sha256_text(self.text)


# --------------------------------------------------------------------------- chunks


class ChunkKind(StrEnum):
    CHUNK = "chunk"  # standard retrievable unit
    PARENT = "parent"  # large unit returned for generation (small-to-big)
    QUESTION = "question"  # hypothetical question pointing to a chunk
    SUMMARY = "summary"  # doc / cluster summary (two-stage or RAPTOR)
    PROPOSITION = "proposition"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_version: str = "1"
    kind: ChunkKind = ChunkKind.CHUNK
    parent_id: str | None = None  # parent chunk (small-to-big) or the chunk a question points to
    position: int = 0  # ordinal within the document (neighbour expansion)
    text: str  # display / citation text (original)
    embed_text: str | None = None  # breadcrumb + context + text, used for embedding and BM25
    title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    token_count: int = 0
    tenant_id: str = "default"
    acl: ACL = Field(default_factory=ACL)
    source_uri: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    enrichment_version: str | None = None

    @property
    def index_text(self) -> str:
        return self.embed_text or self.text

    @staticmethod
    def make_id(doc_id: str, version: str, position: int, kind: str = "chunk", salt: str = "") -> str:
        raw = f"{doc_id}|{version}|{kind}|{position}|{salt}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    source: str = ""  # retriever / stage that produced the score
    stage_scores: dict[str, float] = Field(default_factory=dict)

    def with_score(self, score: float, stage: str) -> ScoredChunk:
        scores = dict(self.stage_scores)
        scores[stage] = score
        return ScoredChunk(chunk=self.chunk, score=score, source=stage, stage_scores=scores)


# --------------------------------------------------------------------------- conversation / answers


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Turn(BaseModel):
    role: Role
    content: str
    created_at: datetime = Field(default_factory=utcnow)


class Citation(BaseModel):
    index: int  # the [n] marker
    chunk_id: str
    doc_id: str
    title: str = ""
    source_uri: str = ""
    page: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    snippet: str = ""
    supported: bool | None = None  # set by citation verification


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"
    CLARIFY = "clarify"
    SMALLTALK = "smalltalk"
    ESCALATED = "escalated"


class Answer(BaseModel):
    text: str
    status: AnswerStatus = AnswerStatus.ANSWERED
    citations: list[Citation] = Field(default_factory=list)
    route: str = ""
    standalone_query: str = ""
    context_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    guardrail_flags: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    trace_id: str = ""
