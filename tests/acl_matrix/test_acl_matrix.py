"""ACL matrix (docs/13 §3): users × documents; no unauthorized chunk may ever appear
in the *retrieved* set (not just the answer), on any route."""

import pytest

from chatbot.core.types import Principal

SECURITY_DOC = "file:security-policy.md"
USERS = {
    "employee": Principal(user_id="alice", groups=frozenset({"employees"})),
    "security": Principal(user_id="sam", groups=frozenset({"employees", "security"})),
    "outsider": Principal(user_id="eve", groups=frozenset({"contractors"})),
    "other_tenant": Principal(
        user_id="tom", groups=frozenset({"employees", "security"}), tenant_id="t2"
    ),
}
QUERIES = [
    "How do I report a security incident?",
    "laptop encryption FileVault BitLocker",
    "MFA hardware security keys production access",
    "refund processing times",
]


@pytest.mark.parametrize("user", list(USERS))
@pytest.mark.parametrize("query", QUERIES)
async def test_no_unauthorized_chunk_is_retrieved(service, user, query):
    p = USERS[user]
    res = await service.rag.retrieve([query], p)
    docs = {h.chunk.doc_id for h in [*res.candidates, *res.reranked, *res.context]}
    if user == "security":
        return
    if user in ("outsider", "other_tenant"):
        assert docs == set(), f"{user} retrieved {docs}"
    else:
        assert SECURITY_DOC not in docs


async def test_security_team_can_see_restricted_doc(service):
    res = await service.rag.retrieve([QUERIES[0]], USERS["security"])
    assert SECURITY_DOC in {h.chunk.doc_id for h in res.context}


async def test_answers_never_cite_unauthorized_docs(service):
    for q in QUERIES:
        answer, _ = await service.rag.chat(q, USERS["employee"])
        assert SECURITY_DOC not in {c.doc_id for c in answer.citations}
