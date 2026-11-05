"""Qdrant vector store (docs/05-indexing-vector-stores.md §2–3).

``qdrant-client`` is imported lazily (``pip install chatbot[qdrant]``). The core
:class:`~chatbot.core.filters.Filter` **plus the principal's tenant and ACL** are
translated into a Qdrant payload filter, so access control is enforced *inside*
the filtered HNSW query (never post-filtered after top-K, docs/07 §6). Payload
indexes are created on every filterable field (§3).

:func:`build_qdrant_filter` returns the Qdrant REST/JSON filter structure as plain
dicts — it is dependency-free and unit-tested; the store validates it into
``models.Filter`` at query time.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import numpy as np

from chatbot.core.filters import _TOP_LEVEL, Condition, Filter, Op
from chatbot.core.types import Chunk, Principal, ScoredChunk
from chatbot.embed.base import EmbeddingInfo, l2_normalize
from chatbot.index.schema import FieldType, IndexSchema, chunk_to_payload, payload_to_chunk

log = logging.getLogger(__name__)

_ID_NAMESPACE = uuid.UUID("6f1c3a52-6a43-4f43-9a51-2f0b5f3b9c11")


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be UUIDs or ints: derive a stable UUIDv5 from ``chunk_id``."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def payload_key(field: str) -> str:
    """Map a filter-DSL field to its payload key (non top-level fields live in metadata)."""
    if field in _TOP_LEVEL or field.startswith(("metadata.", "acl.")):
        return field
    return f"metadata.{field}"


def _jsonable(v: Any) -> Any:
    if isinstance(v, StrEnum):
        return v.value
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, (set, frozenset, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    return v


def _match(key: str, value: Any) -> dict[str, Any]:
    return {"key": key, "match": {"value": _jsonable(value)}}


def _match_any(key: str, values: Any) -> dict[str, Any]:
    return {"key": key, "match": {"any": _jsonable(list(values))}}


def _condition(cond: Condition) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Translate one condition → (must clauses, must_not clauses)."""
    key = payload_key(cond.field)
    v = cond.value
    match cond.op:
        case Op.EQ | Op.CONTAINS:
            if v is None:
                return [{"is_null": {"key": key}}], []
            return [_match(key, v)], []
        case Op.NE:
            return [], [_match(key, v)]
        case Op.IN:
            return [_match_any(key, v)], []
        case Op.NIN:
            return [], [_match_any(key, v)]
        case Op.EXISTS:
            want = True if v is None else bool(v)
            clause = {"is_empty": {"key": key}}
            return ([], [clause]) if want else ([clause], [])
        case Op.GTE | Op.LTE | Op.GT | Op.LT:
            return [{"key": key, "range": {cond.op.value: _jsonable(v)}}], []
    raise ValueError(f"unsupported op {cond.op}")  # pragma: no cover


def acl_clauses(principal: Principal) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tenant isolation + ACL semantics of :meth:`ACL.permits` as Qdrant clauses."""
    groups = sorted(principal.groups)
    allow: list[dict[str, Any]] = [
        {"key": "acl.public", "match": {"value": True}},
        _match("acl.allow_users", principal.user_id),
    ]
    must_not: list[dict[str, Any]] = [_match("acl.deny_users", principal.user_id)]
    if groups:
        allow.append(_match_any("acl.allow_groups", groups))
        must_not.append(_match_any("acl.deny_groups", groups))
    must = [_match("tenant_id", principal.tenant_id), {"should": allow}]
    return must, must_not


def build_qdrant_filter(principal: Principal | None, flt: Filter | None) -> dict[str, Any] | None:
    """Combined ACL + metadata filter as a Qdrant JSON filter (``None`` = no filter)."""
    must: list[dict[str, Any]] = []
    must_not: list[dict[str, Any]] = []
    if principal is not None:
        m, n = acl_clauses(principal)
        must += m
        must_not += n
    if flt is not None:
        for cond in flt.must:
            m, n = _condition(cond)
            must += m
            must_not += n
        for cond in flt.must_not:
            m, n = _condition(cond)
            # NOT(a AND NOT b) for a negated condition: wrap the translated condition.
            must_not.append({"must": m, "must_not": n} if (n or len(m) != 1) else m[0])
    if not must and not must_not:
        return None
    out: dict[str, Any] = {}
    if must:
        out["must"] = must
    if must_not:
        out["must_not"] = must_not
    return out


_SCHEMA_TYPES = {
    FieldType.KEYWORD: "keyword",
    FieldType.INTEGER: "integer",
    FieldType.FLOAT: "float",
    FieldType.BOOL: "bool",
    FieldType.DATETIME: "datetime",
    FieldType.TEXT: "text",
}


class QdrantVectorStore:
    """:class:`~chatbot.core.interfaces.VectorStore` backed by a Qdrant collection or alias."""

    def __init__(
        self,
        schema: IndexSchema,
        *,
        dim: int | None = None,
        url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        collection: str | None = None,
        on_disk_vectors: bool = False,
        name: str = "qdrant",
    ) -> None:
        if dim is None:
            if schema.embedding is None:
                raise ValueError("pass dim or a schema with embedding info")
            dim = schema.embedding.dims
        self.schema = schema
        self.dim = dim
        self.url = url
        self.api_key = api_key
        self.collection = collection or schema.alias_name
        self.on_disk_vectors = on_disk_vectors
        self.name = name
        self._client: Any | None = client

    @property
    def embedding(self) -> EmbeddingInfo | None:
        return self.schema.embedding

    # ------------------------------------------------------------------ client
    @staticmethod
    def _models() -> Any:
        try:
            from qdrant_client import models
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("QdrantVectorStore requires `pip install qdrant-client`") from exc
        return models

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import AsyncQdrantClient
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("QdrantVectorStore requires `pip install qdrant-client`") from exc
            self._client = AsyncQdrantClient(url=self.url or "http://localhost:6333", api_key=self.api_key)
        return self._client

    def _filter(self, principal: Principal | None, flt: Filter | None) -> Any:
        raw = build_qdrant_filter(principal, flt)
        return None if raw is None else self._models().Filter.model_validate(raw)

    # ------------------------------------------------------------------ admin
    async def create_collection(self, schema: IndexSchema | None = None) -> str:
        """Create the versioned physical collection with HNSW config + payload indexes."""
        schema = schema or self.schema
        models = self._models()
        client = self._get_client()
        name = schema.collection_name
        quant = None
        if schema.vector_dtype == "int8":
            quant = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
            )
        elif schema.vector_dtype == "binary":
            quant = models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))
        await client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=self.dim, distance=models.Distance.COSINE, on_disk=self.on_disk_vectors
            ),
            hnsw_config=models.HnswConfigDiff(m=schema.hnsw_m, ef_construct=schema.hnsw_ef_construction),
            quantization_config=quant,
        )
        for f in schema.indexed_fields():
            await client.create_payload_index(
                collection_name=name, field_name=f.name, field_schema=_SCHEMA_TYPES[f.type]
            )
        log.info("created qdrant collection %s", name)
        return name

    async def switch_alias(self, alias: str, collection: str) -> None:
        """Atomically point ``alias`` at ``collection`` (delete + create in one request)."""
        models = self._models()
        ops: list[Any] = [
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)),
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
            ),
        ]
        try:
            await self._get_client().update_collection_aliases(change_aliases_operations=ops)
        except Exception:
            # The alias may not exist yet on first deploy: create only.
            await self._get_client().update_collection_aliases(change_aliases_operations=ops[1:])

    async def drop_collection(self, collection: str) -> None:
        await self._get_client().delete_collection(collection_name=collection)

    # ------------------------------------------------------------------ VectorStore
    async def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        arr = l2_normalize(np.asarray(vectors, dtype=np.float32))
        if arr.shape != (len(chunks), self.dim):
            raise ValueError(f"expected vectors of shape ({len(chunks)}, {self.dim}), got {arr.shape}")
        models = self._models()
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id), vector=v.tolist(), payload=chunk_to_payload(c, self.embedding)
            )
            for c, v in zip(chunks, arr, strict=True)
        ]
        await self._get_client().upsert(collection_name=self.collection, points=points, wait=True)

    async def delete_doc(self, doc_id: str) -> int:
        flt = self._filter(None, Filter.eq(doc_id=doc_id))
        client = self._get_client()
        n = (await client.count(collection_name=self.collection, count_filter=flt, exact=True)).count
        if n:
            await client.delete(
                collection_name=self.collection,
                points_selector=self._models().FilterSelector(filter=flt),
                wait=True,
            )
        return int(n)

    async def search(
        self,
        vector: np.ndarray,
        k: int,
        principal: Principal | None,
        flt: Filter | None = None,
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []
        q = l2_normalize(np.asarray(vector, dtype=np.float32))
        models = self._models()
        resp = await self._get_client().query_points(
            collection_name=self.collection,
            query=q.tolist(),
            query_filter=self._filter(principal, flt),
            limit=k,
            with_payload=True,
            search_params=models.SearchParams(hnsw_ef=max(self.schema.hnsw_ef_search, k)),
        )
        return [
            ScoredChunk(
                chunk=payload_to_chunk(p.payload or {}),
                score=float(p.score),
                source=self.name,
                stage_scores={self.name: float(p.score)},
            )
            for p in resp.points
        ]

    async def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        points = await self._get_client().retrieve(
            collection_name=self.collection, ids=[point_id(c) for c in chunk_ids], with_payload=True
        )
        by_id = {c.chunk_id: c for c in (payload_to_chunk(p.payload or {}) for p in points)}
        return [by_id[c] for c in chunk_ids if c in by_id]

    async def by_doc(self, doc_id: str) -> list[Chunk]:
        flt = self._filter(None, Filter.eq(doc_id=doc_id))
        out: list[Chunk] = []
        offset: Any = None
        client = self._get_client()
        while True:
            points, offset = await client.scroll(
                collection_name=self.collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            out.extend(payload_to_chunk(p.payload or {}) for p in points)
            if offset is None:
                break
        return sorted(out, key=lambda c: (c.kind.value, c.position, c.chunk_id))

    async def count(self) -> int:
        resp = await self._get_client().count(collection_name=self.collection, exact=True)
        return int(resp.count)
