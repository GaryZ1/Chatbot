"""RAG-triad and correctness judges (docs/12-evaluation.md §2 and §4).

* :class:`FaithfulnessJudge` — claim decomposition, then a claim-level verdict
  against the retrieved context ("claim-level faithfulness beats holistic scoring").
* :class:`CorrectnessJudge` — semantic/factual match to the reference answer, both
  binary (preferred, more reliable) and a 1-5 rubric score.
* :class:`AnswerRelevanceJudge` — does the answer address the question?
* :class:`ContextJudge` — context relevance (share of passages relevant to the
  query) and context recall (share of reference statements attributable to context).
* :class:`CitationSupportJudge` — per-statement source support used for ALCE
  citation precision/recall.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from chatbot.evaluation.judges.base import Judge, numbered_contexts, xml_block
from chatbot.evaluation.metrics.generation import ClaimSupport, faithfulness_score, split_statements

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- prompts

CLAIMS_SYSTEM = """You decompose an answer into atomic factual claims.
Rules:
- Each claim is a single, self-contained factual statement (resolve pronouns).
- Skip greetings, hedges, meta statements ("according to the documents") and refusals.
- Do not add facts that the answer does not state.
Return JSON: {"claims": ["...", "..."]}."""

FAITHFULNESS_SYSTEM = """You verify whether each claim is supported by the provided passages.
Rubric per claim:
- "supported": the passages state or directly entail the claim.
- "unsupported": the passages do not contain enough information (even if the claim is true in the world).
- "contradicted": the passages state something incompatible with the claim.
Use only the passages, never outside knowledge. Write your reasoning first, then the verdict.
Return JSON: {"verdicts": [{"claim": "...", "reasoning": "...", "verdict": "supported|unsupported|contradicted"}]}
with exactly one entry per claim, in the given order."""

CORRECTNESS_SYSTEM = """You grade a candidate answer against a reference answer for the same question.
Criteria: factual agreement with the reference on every key point; no contradicting facts;
extra correct detail is fine; missing a key point or stating a wrong number/date/condition is incorrect.
Score rubric (1-5):
5 = fully correct and complete; 4 = correct with a minor omission; 3 = partially correct;
2 = mostly incorrect or missing the main point; 1 = wrong or irrelevant.
"correct" is true only for scores 4-5. Write reasoning first, then the verdict.
Return JSON: {"reasoning": "...", "correct": true|false, "score": 1-5}."""

RELEVANCE_SYSTEM = """You judge whether an answer addresses the user's question (not whether it is true).
Score rubric (1-5): 5 = directly and completely addresses it; 4 = addresses it with minor digressions;
3 = partially addresses it; 2 = mostly off-topic; 1 = does not address it.
A justified refusal ("the documents do not say") counts as relevant (score 3) when the question cannot be answered.
Write reasoning first. Return JSON: {"reasoning": "...", "relevant": true|false, "score": 1-5}
where relevant is true for scores 4-5."""

CONTEXT_RELEVANCE_SYSTEM = """You judge, for each passage, whether it contains information useful to answer the question.
Return JSON: {"passages": [{"id": <passage id>, "reasoning": "...", "relevant": true|false}]}
with one entry per passage."""

CONTEXT_RECALL_SYSTEM = """You check whether each statement of a reference answer can be attributed to the passages.
Return JSON: {"statements": [{"statement": "...", "reasoning": "...", "attributed": true|false}]}
with one entry per given statement, in order."""

CITATION_SYSTEM = """For each statement, decide which of its cited passages support it.
A passage supports a statement when it states or directly entails it.
"jointly_supported" is true when all cited passages together entail the statement.
Return JSON: {"statements": [{"index": <statement index>, "reasoning": "...",
"supporting_ids": [<passage ids that alone support it>], "jointly_supported": true|false}]}"""


# --------------------------------------------------------------------------- schemas


class ClaimList(BaseModel):
    claims: list[str] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    claim: str
    reasoning: str = ""
    verdict: Literal["supported", "unsupported", "contradicted"]


class ClaimVerdicts(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)


class FaithfulnessResult(BaseModel):
    score: float | None
    claims: list[ClaimVerdict] = Field(default_factory=list)

    @property
    def unsupported(self) -> list[str]:
        return [c.claim for c in self.claims if c.verdict != "supported"]


class CorrectnessVerdict(BaseModel):
    reasoning: str = ""
    correct: bool
    score: int = Field(ge=1, le=5)


class RelevanceVerdict(BaseModel):
    reasoning: str = ""
    relevant: bool
    score: int = Field(ge=1, le=5)


class PassageRelevance(BaseModel):
    id: int
    reasoning: str = ""
    relevant: bool


class PassageRelevanceList(BaseModel):
    passages: list[PassageRelevance] = Field(default_factory=list)


class StatementAttribution(BaseModel):
    statement: str
    reasoning: str = ""
    attributed: bool


class StatementAttributionList(BaseModel):
    statements: list[StatementAttribution] = Field(default_factory=list)


class StatementSupport(BaseModel):
    index: int
    reasoning: str = ""
    supporting_ids: list[int] = Field(default_factory=list)
    jointly_supported: bool = False


class StatementSupportList(BaseModel):
    statements: list[StatementSupport] = Field(default_factory=list)


# --------------------------------------------------------------------------- judges


class FaithfulnessJudge(Judge):
    """Two-step claim-level faithfulness: decompose → verify each claim."""

    async def claims(self, question: str, answer: str) -> list[str] | None:
        user = "\n\n".join([xml_block("question", question), xml_block("answer", answer)])
        res = await self._parse(CLAIMS_SYSTEM, user, ClaimList)
        if res is None:
            return None
        return [c.strip() for c in res.claims if c.strip()]

    async def judge(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> FaithfulnessResult | None:
        claims = await self.claims(question, answer)
        if claims is None:
            return None
        if not claims:
            return FaithfulnessResult(score=None, claims=[])
        if not contexts:
            verdicts = [
                ClaimVerdict(claim=c, reasoning="no context", verdict="unsupported") for c in claims
            ]
            return FaithfulnessResult(score=0.0, claims=verdicts)
        listing = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
        user = "\n\n".join(
            [
                xml_block("passages", numbered_contexts(contexts)),
                xml_block("claims", listing),
            ]
        )
        res = await self._parse(FAITHFULNESS_SYSTEM, user, ClaimVerdicts)
        if res is None:
            return None
        verdicts = list(res.verdicts)
        if len(verdicts) != len(claims):
            logger.warning(
                "faithfulness judge returned %d verdicts for %d claims", len(verdicts), len(claims)
            )
            # Missing verdicts are conservatively treated as unsupported.
            verdicts = verdicts[: len(claims)]
            verdicts += [
                ClaimVerdict(claim=c, reasoning="no verdict returned", verdict="unsupported")
                for c in claims[len(verdicts) :]
            ]
        score = faithfulness_score([v.verdict == "supported" for v in verdicts])
        return FaithfulnessResult(score=score, claims=verdicts)


class CorrectnessJudge(Judge):
    async def judge(self, question: str, answer: str, reference: str) -> CorrectnessVerdict | None:
        user = "\n\n".join(
            [
                xml_block("question", question),
                xml_block("reference_answer", reference),
                xml_block("candidate_answer", answer),
            ]
        )
        return await self._parse(CORRECTNESS_SYSTEM, user, CorrectnessVerdict)


class AnswerRelevanceJudge(Judge):
    async def judge(self, question: str, answer: str) -> RelevanceVerdict | None:
        user = "\n\n".join([xml_block("question", question), xml_block("answer", answer)])
        return await self._parse(RELEVANCE_SYSTEM, user, RelevanceVerdict)


class ContextJudge(Judge):
    async def relevance(self, question: str, contexts: Sequence[str]) -> float | None:
        """Share of passages relevant to the question (context precision)."""
        if not contexts:
            return None
        user = "\n\n".join(
            [xml_block("question", question), xml_block("passages", numbered_contexts(contexts))]
        )
        res = await self._parse(CONTEXT_RELEVANCE_SYSTEM, user, PassageRelevanceList)
        if res is None:
            return None
        by_id = {p.id: p.relevant for p in res.passages}
        return sum(by_id.get(i, False) for i in range(1, len(contexts) + 1)) / len(contexts)

    async def recall(self, question: str, reference: str, contexts: Sequence[str]) -> float | None:
        """Share of reference-answer statements attributable to the context."""
        statements = [s.text for s in split_statements(reference)]
        if not statements:
            return None
        if not contexts:
            return 0.0
        listing = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, start=1))
        user = "\n\n".join(
            [
                xml_block("question", question),
                xml_block("passages", numbered_contexts(contexts)),
                xml_block("reference_statements", listing),
            ]
        )
        res = await self._parse(CONTEXT_RECALL_SYSTEM, user, StatementAttributionList)
        if res is None:
            return None
        flags = [s.attributed for s in res.statements[: len(statements)]]
        flags += [False] * (len(statements) - len(flags))
        return sum(flags) / len(statements)


class CitationSupportJudge(Judge):
    """Produces :class:`ClaimSupport` rows for ALCE scoring from an answer with ``[n]``
    markers and the passages those markers refer to."""

    async def judge(self, answer: str, sources: Mapping[int, str]) -> list[ClaimSupport] | None:
        statements = split_statements(answer)
        if not statements:
            return []
        lines = []
        for i, st in enumerate(statements, start=1):
            cited = ", ".join(str(m) for m in st.markers) or "none"
            lines.append(f"{i}. {st.text} (cites: {cited})")
        used = sorted({m for st in statements for m in st.markers if m in sources})
        passages = "\n".join(xml_block("passage", sources[m], id=m) for m in used)
        user = "\n\n".join(
            [xml_block("passages", passages), xml_block("statements", "\n".join(lines))]
        )
        res = await self._parse(CITATION_SYSTEM, user, StatementSupportList)
        if res is None:
            return None
        by_idx = {s.index: s for s in res.statements}
        out: list[ClaimSupport] = []
        for i, st in enumerate(statements, start=1):
            verdict = by_idx.get(i)
            cited_ids = [str(m) for m in st.markers]
            if verdict is None or not st.markers:
                out.append(ClaimSupport(claim=st.text, cited_ids=cited_ids, jointly_supported=False))
                continue
            supporting = {str(m) for m in verdict.supporting_ids if m in st.markers}
            out.append(
                ClaimSupport(
                    claim=st.text,
                    cited_ids=cited_ids,
                    supporting_ids=supporting,
                    jointly_supported=verdict.jointly_supported or bool(supporting),
                )
            )
        return out
