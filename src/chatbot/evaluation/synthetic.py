"""Synthetic golden-set generation (docs/12-evaluation.md §3.1, "Synthetic").

Evolution-style generation in the spirit of RAGAS ``TestsetGenerator``:

* ``simple`` — a single-fact lookup answerable from one chunk;
* ``reasoning`` — needs an inference step over one chunk (conditions, comparisons, arithmetic);
* ``multi_context`` — needs facts from two chunks (multi-hop / comparison);
* ``conditional`` — adds a condition or scenario ("if I am a contractor ...").

Every candidate then goes through a judge filter for *answerability* from the source
chunks, *standalone-ness* and *realism* (would a real user ask this?). Humans
should still review a sample before promotion to the golden set.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, Field

from chatbot.core.types import Chunk, ChunkKind
from chatbot.evaluation.dataset import Difficulty, ExampleSource, GoldenExample
from chatbot.evaluation.judges.base import UNTRUSTED_NOTE, xml_block
from chatbot.llm.base import LLM, LLMError, LLMRequest, Tier

logger = logging.getLogger(__name__)


class Evolution(StrEnum):
    SIMPLE = "simple"
    REASONING = "reasoning"
    MULTI_CONTEXT = "multi_context"
    CONDITIONAL = "conditional"


#: Default style mix (RAGAS-like: half simple, the rest harder).
DEFAULT_DISTRIBUTION: dict[Evolution, float] = {
    Evolution.SIMPLE: 0.5,
    Evolution.REASONING: 0.2,
    Evolution.MULTI_CONTEXT: 0.15,
    Evolution.CONDITIONAL: 0.15,
}

_DIFFICULTY = {
    Evolution.SIMPLE: Difficulty.EASY,
    Evolution.REASONING: Difficulty.MEDIUM,
    Evolution.MULTI_CONTEXT: Difficulty.HARD,
    Evolution.CONDITIONAL: Difficulty.MEDIUM,
}

GENERATOR_SYSTEM = f"""You write evaluation questions for a company knowledge-base assistant.
Write ONE question a real employee or customer would plausibly ask, plus a concise reference answer
that is fully supported by the given passages. The question must be standalone (no "according to the
passage", no "this document"), must not copy long phrases verbatim, and must be answerable ONLY from the passages.
Return JSON: {{"question": "...", "answer": "...", "intent": "lookup|procedure|comparison|eligibility|numeric|other"}}.
{UNTRUSTED_NOTE}"""

STYLE_INSTRUCTIONS: dict[Evolution, str] = {
    Evolution.SIMPLE: "Style: a direct single-fact lookup answerable from one passage.",
    Evolution.REASONING: (
        "Style: requires one reasoning step over the passage (apply a rule, compare two values, "
        "compute a date or amount). The answer must still be derivable from the passage."
    ),
    Evolution.MULTI_CONTEXT: (
        "Style: requires combining facts from BOTH passages; neither passage alone is enough."
    ),
    Evolution.CONDITIONAL: (
        "Style: include a realistic condition or scenario (role, location, amount, timing) "
        "that changes which rule in the passage applies."
    ),
}

FILTER_SYSTEM = f"""You review a synthetic evaluation question before it enters a golden test set.
Check: (1) answerable: the passages fully support the reference answer; (2) standalone: the question
makes sense without seeing the passages; (3) realism 1-5: how plausible it is that a real user asks this
(5 = very common real question, 1 = artificial or trivia about the document itself).
Write your reasoning first. Return JSON:
{{"reasoning": "...", "answerable": true|false, "standalone": true|false, "realism": 1-5}}.
{UNTRUSTED_NOTE}"""


class SyntheticQA(BaseModel):
    question: str = Field(min_length=5)
    answer: str = Field(min_length=1)
    intent: str = "lookup"


class FilterVerdict(BaseModel):
    reasoning: str = ""
    answerable: bool
    standalone: bool
    realism: int = Field(ge=1, le=5)


class SyntheticConfig(BaseModel):
    tier: Tier = Tier.MID
    judge_tier: Tier = Tier.MID
    min_realism: int = Field(default=3, ge=1, le=5)
    max_concurrency: int = Field(default=4, ge=1)
    seed: int = 0
    #: Chunks shorter than this (characters) carry too little content to ask about.
    min_chunk_chars: int = 200


class SyntheticGenerator:
    def __init__(
        self, llm: LLM, judge_llm: LLM | None = None, config: SyntheticConfig | None = None
    ) -> None:
        self.llm = llm
        self.judge_llm = judge_llm or llm
        self.config = config or SyntheticConfig()
        self._sem = asyncio.Semaphore(self.config.max_concurrency)

    def _plan(
        self, chunks: Sequence[Chunk], n: int, distribution: Mapping[Evolution, float]
    ) -> list[tuple[Evolution, list[Chunk]]]:
        rng = random.Random(self.config.seed)
        pool = sorted(
            (
                c
                for c in chunks
                if c.kind == ChunkKind.CHUNK and len(c.text.strip()) >= self.config.min_chunk_chars
            ),
            key=lambda c: c.chunk_id,
        )
        if not pool:
            return []
        total = sum(distribution.values())
        if total <= 0:
            raise ValueError("distribution weights must sum to > 0")
        styles = list(distribution)
        weights = [distribution[s] / total for s in styles]
        by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for c in pool:
            by_doc[c.doc_id].append(c)
        plan: list[tuple[Evolution, list[Chunk]]] = []
        for _ in range(n):
            style = rng.choices(styles, weights)[0]
            first = rng.choice(pool)
            if style is Evolution.MULTI_CONTEXT:
                others = [c for c in pool if c.chunk_id != first.chunk_id]
                if not others:
                    style = Evolution.SIMPLE
                    plan.append((style, [first]))
                    continue
                # Prefer a different document (cross-doc hop), else another chunk of the same one.
                cross = [c for c in others if c.doc_id != first.doc_id]
                second = rng.choice(cross or others)
                plan.append((style, [first, second]))
            else:
                plan.append((style, [first]))
        return plan

    async def _generate_one(self, style: Evolution, chunks: list[Chunk]) -> GoldenExample | None:
        passages = "\n".join(
            xml_block("passage", c.text, id=i, title=c.title or c.doc_id)
            for i, c in enumerate(chunks, start=1)
        )
        user = f"{STYLE_INSTRUCTIONS[style]}\n\n{xml_block('passages', passages)}"
        async with self._sem:
            try:
                qa = await self.llm.parse(
                    LLMRequest.simple(user, system=GENERATOR_SYSTEM, tier=self.config.tier),
                    SyntheticQA,
                )
            except LLMError as exc:
                logger.warning("synthetic generation failed (%s): %s", style, exc)
                return None
            filter_user = "\n\n".join(
                [
                    xml_block("passages", passages),
                    xml_block("question", qa.question),
                    xml_block("reference_answer", qa.answer),
                ]
            )
            try:
                verdict = await self.judge_llm.parse(
                    LLMRequest.simple(
                        filter_user, system=FILTER_SYSTEM, tier=self.config.judge_tier
                    ),
                    FilterVerdict,
                )
            except LLMError as exc:
                logger.warning("synthetic filter failed, dropping candidate: %s", exc)
                return None
        if not (
            verdict.answerable and verdict.standalone and verdict.realism >= self.config.min_realism
        ):
            logger.debug("filtered synthetic question %r: %s", qa.question, verdict.reasoning)
            return None
        qid = "syn-" + hashlib.sha256(qa.question.strip().lower().encode()).hexdigest()[:12]
        return GoldenExample(
            id=qid,
            question=qa.question.strip(),
            reference_answer=qa.answer.strip(),
            gold_doc_ids=list(dict.fromkeys(c.doc_id for c in chunks)),
            gold_chunk_ids=[c.chunk_id for c in chunks],
            intent=qa.intent or "lookup",
            difficulty=_DIFFICULTY[style],
            must_cite=True,
            should_abstain=False,
            slices=["synthetic", f"evolution:{style.value}"]
            + (["multi_hop"] if style is Evolution.MULTI_CONTEXT else []),
            source=ExampleSource.SYNTHETIC,
            metadata={"evolution": style.value, "realism": verdict.realism, "needs_review": True},
        )

    async def generate(
        self,
        chunks: Sequence[Chunk],
        n: int,
        distribution: Mapping[Evolution, float] | None = None,
    ) -> list[GoldenExample]:
        """Generate up to ``n`` filtered examples (fewer when the filter rejects some).
        Duplicate questions are dropped."""
        if n <= 0:
            return []
        plan = self._plan(chunks, n, distribution or DEFAULT_DISTRIBUTION)
        results = await asyncio.gather(*(self._generate_one(style, cs) for style, cs in plan))
        out: dict[str, GoldenExample] = {}
        for ex in results:
            if ex is not None and ex.id not in out:
                out[ex.id] = ex
        logger.info("synthetic: %d/%d candidates kept", len(out), len(plan))
        return list(out.values())
