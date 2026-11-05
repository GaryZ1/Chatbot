"""Index schema and collection naming (docs/05-indexing-vector-stores.md §3).

``collection: chunks_v3 (alias: chunks_live)`` — physical collections are
versioned; readers only ever address the alias, so a re-index is a blue/green
alias switch (§3 "Aliases for zero-downtime re-indexing"). Every vector carries
the embedding model identity (docs/04-embeddings.md §8).
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from chatbot.core.types import Chunk, utcnow
from chatbot.embed.base import EmbeddingInfo


class FieldType(StrEnum):
    KEYWORD = "keyword"
    INTEGER = "integer"
    FLOAT = "float"
    BOOL = "bool"
    DATETIME = "datetime"
    TEXT = "text"


class PayloadField(BaseModel):
    name: str  # payload key path (dots address nested keys)
    type: FieldType
    indexed: bool = True  # create a payload index (§3: index every field you filter by)


#: Payload layout from §3. Top-level keys mirror :class:`Chunk`; free metadata lives
#: under ``metadata.*`` (the filter DSL reads non top-level fields from metadata).
DEFAULT_PAYLOAD_FIELDS: list[PayloadField] = [
    PayloadField(name="chunk_id", type=FieldType.KEYWORD),
    PayloadField(name="doc_id", type=FieldType.KEYWORD),
    PayloadField(name="doc_version", type=FieldType.KEYWORD),
    PayloadField(name="parent_id", type=FieldType.KEYWORD),
    PayloadField(name="kind", type=FieldType.KEYWORD),
    PayloadField(name="position", type=FieldType.INTEGER),
    PayloadField(name="page", type=FieldType.INTEGER),
    PayloadField(name="title", type=FieldType.KEYWORD),
    PayloadField(name="text", type=FieldType.TEXT, indexed=False),
    PayloadField(name="embed_text", type=FieldType.TEXT, indexed=False),
    PayloadField(name="heading_path", type=FieldType.KEYWORD, indexed=False),
    PayloadField(name="tenant_id", type=FieldType.KEYWORD),
    PayloadField(name="acl.public", type=FieldType.BOOL),
    PayloadField(name="acl.allow_users", type=FieldType.KEYWORD),
    PayloadField(name="acl.allow_groups", type=FieldType.KEYWORD),
    PayloadField(name="acl.deny_users", type=FieldType.KEYWORD),
    PayloadField(name="acl.deny_groups", type=FieldType.KEYWORD),
    PayloadField(name="metadata.classification", type=FieldType.KEYWORD),
    PayloadField(name="metadata.doc_type", type=FieldType.KEYWORD),
    PayloadField(name="metadata.product", type=FieldType.KEYWORD),
    PayloadField(name="metadata.language", type=FieldType.KEYWORD),
    PayloadField(name="metadata.effective_date", type=FieldType.DATETIME),
    PayloadField(name="metadata.updated_at", type=FieldType.DATETIME),
    PayloadField(name="metadata.is_latest", type=FieldType.BOOL),
    PayloadField(name="metadata.quality_score", type=FieldType.FLOAT),
    PayloadField(name="embedding_model", type=FieldType.KEYWORD),
    PayloadField(name="enrichment_version", type=FieldType.KEYWORD),
]

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


class IndexSchema(BaseModel):
    """Logical index definition: ``{base}_v{version}`` behind alias ``{base}_live``."""

    base_name: str = "chunks"
    version: int = 1
    alias: str | None = None
    embedding: EmbeddingInfo | None = None
    vector_dtype: str = "float32"  # float32 | int8 | binary (with float rescoring)
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128
    tenant: str | None = None  # set for per-tenant collections (strong isolation)
    payload_fields: list[PayloadField] = Field(default_factory=lambda: list(DEFAULT_PAYLOAD_FIELDS))

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        if not _NAME_RE.match(self.base_name):
            raise ValueError(f"invalid base_name {self.base_name!r}")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        if self.tenant is not None and not re.match(r"^[A-Za-z0-9_-]{1,40}$", self.tenant):
            raise ValueError(f"invalid tenant {self.tenant!r}")

    @property
    def _prefix(self) -> str:
        return f"{self.base_name}_{self.tenant.lower()}" if self.tenant else self.base_name

    @property
    def collection_name(self) -> str:
        return f"{self._prefix}_v{self.version}"

    @property
    def alias_name(self) -> str:
        return self.alias or f"{self._prefix}_live"

    def next_version(self, embedding: EmbeddingInfo | None = None) -> IndexSchema:
        """Schema for the green collection of a blue/green migration."""
        return self.model_copy(
            update={"version": self.version + 1, "embedding": embedding or self.embedding}
        )

    def indexed_fields(self) -> list[PayloadField]:
        return [f for f in self.payload_fields if f.indexed]


def chunk_to_payload(chunk: Chunk, embedding: EmbeddingInfo | None = None) -> dict[str, Any]:
    """Serialise a chunk to a JSON payload (nested ``acl`` and ``metadata`` keys)."""
    payload = chunk.model_dump(mode="json")
    if embedding is not None:
        payload["embedding_model"] = embedding.key
    payload["indexed_at"] = utcnow().isoformat()
    return payload


_CHUNK_FIELDS = set(Chunk.model_fields)


def payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    return Chunk.model_validate({k: v for k, v in payload.items() if k in _CHUNK_FIELDS})
