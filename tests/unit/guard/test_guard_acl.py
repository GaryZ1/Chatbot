import pytest

from chatbot.core.filters import visible
from chatbot.core.types import ACL, Chunk, Principal, ScoredChunk
from chatbot.guard.acl import (
    AclMatrix,
    LeakError,
    StaticGroupResolver,
    acl_filter,
    assert_no_leak,
    inherit_acl,
)

ALICE = Principal(user_id="alice", groups=frozenset({"eng"}))
BOB = Principal(user_id="bob", groups=frozenset({"sales"}))
EVE = Principal(user_id="eve", groups=frozenset({"eng"}), tenant_id="other")
CAROL = Principal(user_id="carol", groups=frozenset({"eng", "contractors"}))

CHUNKS = [
    Chunk(chunk_id="pub", doc_id="handbook", text="t", acl=ACL(public=True)),
    Chunk(chunk_id="eng1", doc_id="eng-wiki", text="t", acl=ACL(allow_groups=["eng"], deny_groups=["contractors"])),
    Chunk(chunk_id="sales1", doc_id="pipeline", text="t", acl=ACL(allow_groups=["sales"])),
    Chunk(chunk_id="bob1", doc_id="bob-notes", text="t", acl=ACL(allow_users=["bob"])),
]


def test_assert_no_leak():
    assert_no_leak([ScoredChunk(chunk=CHUNKS[0], score=1.0)], ALICE)
    with pytest.raises(LeakError) as e:
        assert_no_leak([CHUNKS[1], CHUNKS[2]], ALICE)
    assert e.value.chunk_ids == ["sales1"]
    with pytest.raises(LeakError):
        assert_no_leak([CHUNKS[0]], EVE)  # other tenant


def test_acl_filter():
    assert [c.chunk_id for c in acl_filter(CHUNKS, CAROL)] == ["pub"]  # deny wins


def test_matrix_expectations():
    m = AclMatrix([ALICE, BOB, CAROL, EVE], CHUNKS)
    exp = m.expected()
    assert exp[("alice", "eng-wiki")] and not exp[("alice", "pipeline")]
    assert exp[("bob", "bob-notes")] and exp[("bob", "pipeline")]
    assert not exp[("carol", "eng-wiki")]
    assert not any(v for (u, _), v in exp.items() if u == "eve")
    assert "alice" in m.render()


async def test_matrix_check_correct_and_leaky_retrievers():
    m = AclMatrix([ALICE, BOB, CAROL, EVE], CHUNKS)

    async def correct(p):
        return [ScoredChunk(chunk=c, score=1.0) for c in CHUNKS if visible(c, p)]

    async def leaky(p):  # forgets deny rules and tenant isolation
        return [c for c in CHUNKS if c.acl.public or p.groups & set(c.acl.allow_groups)]

    assert await m.check(correct, require_complete=True) == []
    await m.assert_ok(correct)
    violations = await m.check(leaky)
    assert {(v.user_id, v.doc_id) for v in violations} == {("carol", "eng-wiki"), ("eve", "handbook"), ("eve", "eng-wiki")}
    with pytest.raises(AssertionError):
        await m.assert_ok(leaky)

    async def over_filtering(p):
        return []

    missing = await m.check(over_filtering, require_complete=True)
    assert all(v.kind == "missing" for v in missing) and missing


async def test_static_resolver_ttl_and_nested_groups():
    now = [0.0]
    r = StaticGroupResolver(
        {"alice": {"eng-backend"}}, group_parents={"eng-backend": {"eng"}}, ttl_s=60, clock=lambda: now[0]
    )
    p = await r.resolve("alice")
    assert p.groups == {"eng-backend", "eng"}
    r.set_memberships("alice", {"sales"})
    assert (await r.resolve("alice")).groups == {"eng-backend", "eng"}  # cached
    now[0] = 61
    assert (await r.resolve("alice")).groups == {"sales"}
    assert r.lookups == 2


def test_inherit_acl():
    a = inherit_acl(ACL(allow_groups=["eng"]), ACL(allow_users=["x"], deny_users=["y"]))
    assert a.allow_groups == ["eng"] and a.allow_users == ["x"] and a.deny_users == ["y"]
