import json

from agent_fakes import CORPUS, FakeRetriever

from chatbot.agent import CorrectiveRAG, CRAGConfig
from chatbot.agent.crag import State
from chatbot.core.types import AnswerStatus, Principal
from chatbot.llm.fake import FakeLLM, last_user_text

ALICE = Principal(user_id="alice", groups=frozenset({"staff"}))


def make_llm(grounded_seq=(True,)):
    seq = list(grounded_seq)

    def handler(req):
        user = last_user_text(req)
        if "grade whether a retrieved passage" in req.system:
            return json.dumps(
                {"reasoning": "r", "grade": "correct" if "full refund" in user else "incorrect"}
            )
        if "rewrite a search query" in req.system:
            return json.dumps({"reasoning": "r", "query": "refund purchase days"})
        if "fully supported" in req.system:
            g = seq.pop(0) if seq else True
            return json.dumps(
                {"reasoning": "r", "grounded": g, "unsupported_claims": [] if g else ["x"]}
            )
        return "{}"

    return FakeLLM(handler=handler)


async def gen(question, chunks, feedback):
    return f"answer from {len(chunks)} chunks" + (" (revised)" if feedback else "")


async def test_crag_retries_then_answers():
    llm = make_llm()
    retr = FakeRetriever(CORPUS)
    crag = CorrectiveRAG(llm, retr, gen, config=CRAGConfig(k=3, max_rewrites=2))
    res = await crag.run(
        "PTO employees", ALICE
    )  # first retrieval only finds the PTO chunk → incorrect
    states = [e.state for e in res.trace]
    assert State.REWRITE in states and states.count(State.RETRIEVE) == 2
    assert res.status is AnswerStatus.ANSWERED and res.grounded is True
    assert res.queries == ["PTO employees", "refund purchase days"]
    assert any(c.chunk.chunk_id == "r1" for c in res.chunks)
    assert all(p == ALICE for _, p in retr.calls)
    assert res.retrieval_calls <= crag.worst_case_calls()["retrieval"]
    assert res.llm_calls <= crag.worst_case_calls()["llm_total"]


async def test_crag_abstains_when_nothing_relevant():
    crag = CorrectiveRAG(make_llm(), FakeRetriever(CORPUS), gen, config=CRAGConfig(max_rewrites=0))
    res = await crag.run("PTO employees", ALICE)
    assert res.status is AnswerStatus.ABSTAINED and res.chunks == []


async def test_crag_regenerates_once_then_abstains():
    crag = CorrectiveRAG(make_llm(grounded_seq=(False, False)), FakeRetriever(CORPUS), gen)
    res = await crag.run("refund within 30 days purchase", ALICE)
    states = [e.state for e in res.trace]
    assert states.count(State.REGENERATE) == 1 and states.count(State.CHECK) == 2
    assert res.status is AnswerStatus.ABSTAINED

    crag2 = CorrectiveRAG(make_llm(grounded_seq=(False, True)), FakeRetriever(CORPUS), gen)
    res2 = await crag2.run("refund within 30 days purchase", ALICE)
    assert res2.status is AnswerStatus.ANSWERED and res2.answer.endswith("(revised)")


async def test_crag_fallback_retriever():
    class Fallback(FakeRetriever):
        name = "web"

    fb = Fallback(CORPUS[:1])
    llm = make_llm()
    crag = CorrectiveRAG(
        llm,
        FakeRetriever(CORPUS[2:3]),
        gen,
        fallback_retriever=fb,
        config=CRAGConfig(max_rewrites=0),
    )
    res = await crag.run("refund 30 days policy", ALICE)
    assert State.FALLBACK in [e.state for e in res.trace]
    assert res.status is AnswerStatus.ANSWERED
