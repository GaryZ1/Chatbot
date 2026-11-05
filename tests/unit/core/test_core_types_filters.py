from datetime import UTC, datetime

from chatbot.core import ACL, Chunk, Condition, Filter, Op, Principal, visible
from chatbot.core.failures import FailureCode, diagnose
from chatbot.core.tokens import RegexTokenizer, truncate_to_tokens
from chatbot.core.types import Block, BlockType, Document, sha256_text


def _chunk(**kw):
    base = dict(chunk_id="c1", doc_id="d1", text="hello", tenant_id="t1", acl=ACL(allow_groups=["eng"]))
    base.update(kw)
    return Chunk(**base)


def test_acl_deny_wins_over_allow():
    acl = ACL(allow_groups=["eng"], deny_users=["mallory"])
    assert acl.permits(Principal(user_id="alice", groups=frozenset({"eng"})))
    assert not acl.permits(Principal(user_id="mallory", groups=frozenset({"eng"})))
    assert not acl.permits(Principal(user_id="bob", groups=frozenset({"sales"})))
    assert ACL(public=True).permits(Principal(user_id="anyone"))
    assert not ACL(public=True, deny_groups=["contractors"]).permits(
        Principal(user_id="x", groups=frozenset({"contractors"}))
    )


def test_visible_enforces_tenant_then_acl_then_metadata():
    c = _chunk(metadata={"region": "EMEA", "updated_at": "2026-02-01"})
    alice = Principal(user_id="alice", groups=frozenset({"eng"}), tenant_id="t1")
    assert visible(c, alice)
    assert not visible(c, alice.model_copy(update={"tenant_id": "t2"}))
    assert not visible(c, Principal(user_id="bob", tenant_id="t1"))
    assert visible(c, alice, Filter.eq(region="EMEA"))
    assert not visible(c, alice, Filter.eq(region="AMER"))
    newer = Filter(must=[Condition(field="updated_at", op=Op.GTE, value="2026-01-01")])
    assert visible(c, alice, newer)
    assert visible(c, None)  # trusted offline job


def test_filter_ops_on_lists_and_missing_fields():
    c = _chunk(metadata={"product": ["a", "b"]})
    assert Filter(must=[Condition(field="product", op=Op.CONTAINS, value="a")]).matches(c)
    assert Filter(must=[Condition(field="product", op=Op.IN, value=["b", "z"])]).matches(c)
    assert Filter(must_not=[Condition(field="product", op=Op.IN, value=["z"])]).matches(c)
    assert not Filter(must=[Condition(field="missing", op=Op.GTE, value=3)]).matches(c)
    dt = Filter(must=[Condition(field="ts", op=Op.LT, value=datetime(2027, 1, 1, tzinfo=UTC))])
    assert dt.matches(_chunk(metadata={"ts": "2026-05-01T00:00:00+00:00"}))


def test_scope_key_differs_by_groups():
    a = Principal(user_id="u", groups=frozenset({"x"}))
    b = Principal(user_id="u", groups=frozenset({"x", "y"}))
    assert a.scope_key() != b.scope_key()
    assert a.scope_key() == Principal(user_id="u", groups=frozenset({"x"})).scope_key()


def test_document_hash_normalises_whitespace():
    d1 = Document(doc_id="d", blocks=[Block(type=BlockType.PARAGRAPH, text="a  b\n c")])
    d2 = Document(doc_id="d", blocks=[Block(type=BlockType.PARAGRAPH, text="a b c")])
    assert d1.content_hash == d2.content_hash == sha256_text("a b c")


def test_tokenizer_spans_and_truncate():
    tok = RegexTokenizer(piece_len=4)
    text = "Refunds within 30 days, abcdefghij."
    spans = tok.spans(text)
    assert all(0 <= s < e <= len(text) for s, e in spans)
    assert tok.count(text) == len(spans)
    assert truncate_to_tokens(text, 1, tok) == "Refu"
    assert len(tok.spans(truncate_to_tokens(text, 3, tok))) == 3


def test_diagnose_funnel():
    assert diagnose(["d1"], ["d2"], ["d2"], ["d2"]) is FailureCode.F5_MISSED_TOP_K
    assert diagnose(["d1"], ["d1"], ["d1"], []) is FailureCode.F7_NOT_IN_CONTEXT
    assert diagnose(["d1"], ["d1"], ["d1"], ["d1"]) is None
    assert diagnose(["dx"], [], [], [], corpus_doc_ids={"d1"}) is FailureCode.F1_MISSING_CONTENT
