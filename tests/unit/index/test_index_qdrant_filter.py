from __future__ import annotations

from chatbot.core.filters import Condition, Filter, Op
from chatbot.core.types import ACL, Chunk, Principal
from chatbot.index import (
    IndexSchema,
    build_qdrant_filter,
    chunk_to_payload,
    payload_to_chunk,
    point_id,
)


def test_acl_translated_into_query_filter() -> None:
    p = Principal(user_id="alice", groups=frozenset({"eng", "all"}), tenant_id="t1")
    f = build_qdrant_filter(p, Filter(must=[Condition(field="doc_type", op=Op.IN, value=["faq"])]))
    assert f is not None
    must, must_not = f["must"], f["must_not"]
    assert {"key": "tenant_id", "match": {"value": "t1"}} in must
    should = next(m["should"] for m in must if "should" in m)
    assert {"key": "acl.public", "match": {"value": True}} in should
    assert {"key": "acl.allow_groups", "match": {"any": ["all", "eng"]}} in should
    assert {"key": "acl.deny_users", "match": {"value": "alice"}} in must_not
    assert {"key": "metadata.doc_type", "match": {"any": ["faq"]}} in must


def test_filter_ops_and_trusted() -> None:
    assert build_qdrant_filter(None, None) is None
    f = build_qdrant_filter(
        None,
        Filter(
            must=[
                Condition(field="updated_at", op=Op.GTE, value="2026-01-01"),
                Condition(field="doc_id", op=Op.NE, value="x"),
            ],
            must_not=[Condition(field="is_latest", op=Op.EQ, value=False)],
        ),
    )
    assert f == {
        "must": [{"key": "metadata.updated_at", "range": {"gte": "2026-01-01"}}],
        "must_not": [
            {"key": "doc_id", "match": {"value": "x"}},
            {"key": "metadata.is_latest", "match": {"value": False}},
        ],
    }


def test_schema_and_payload_roundtrip() -> None:
    s = IndexSchema(version=3)
    assert s.collection_name == "chunks_v3" and s.alias_name == "chunks_live"
    assert s.next_version().collection_name == "chunks_v4"
    assert IndexSchema(tenant="Acme").collection_name == "chunks_acme_v1"
    ch = Chunk(chunk_id="c1", doc_id="d", text="t", acl=ACL(allow_groups=["g"]), metadata={"x": 1})
    assert payload_to_chunk(chunk_to_payload(ch)) == ch
    assert point_id("c1") == point_id("c1") != point_id("c2")
