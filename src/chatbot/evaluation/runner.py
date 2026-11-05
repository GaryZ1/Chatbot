"""Offline evaluation harness (docs/12-evaluation.md §1, §5, §7).

The system under test is any async callable ``(GoldenExample) -> EvalTrace``; the
trace exposes every stage of the retrieval funnel (first-stage candidates → final
top-k after fusion/rerank → what actually entered the context window), the answer,
citations, per-stage latency and cost. From it the runner computes:

* retrieval metrics at doc or chunk level (Recall@k, Precision@k, Hit@k, MRR, nDCG@k);
* lexical answer metrics, abstention accuracy, citation presence;
* optional LLM-judge metrics (faithfulness, correctness, relevance, context, ALCE citations);
* the retrieval funnel and an F-code per failure via :func:`chatbot.core.failures.diagnose`;
* per-slice aggregates, p50/p95 latency per stage and $/query.

Examples run concurrently under a semaphore, each with a timeout; a crashing
example is recorded as an error, never aborting the run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from chatbot.core.failures import FAILURE_STAGE, FailureCode, diagnose
from chatbot.core.types import Answer, AnswerStatus, Chunk, Citation, ScoredChunk, utcnow
from chatbot.evaluation.dataset import GoldenExample, GoldenSet
from chatbot.evaluation.judges.base import JudgeConfig
from chatbot.evaluation.judges.rag import (
    AnswerRelevanceJudge,
    CitationSupportJudge,
    ContextJudge,
    CorrectnessJudge,
    FaithfulnessJudge,
)
from chatbot.evaluation.metrics.generation import (
    abstention_report,
    citation_scores,
    cited_gold_precision,
    exact_match,
    is_abstention,
    token_f1,
    token_recall,
)
from chatbot.evaluation.metrics.retrieval import DEFAULT_KS, dedupe, recall_at_k, retrieval_metrics
from chatbot.evaluation.stats import percentile
from chatbot.llm.base import LLM

logger = logging.getLogger(__name__)

#: Statuses that count as declining to answer.
ABSTAIN_STATUSES = {AnswerStatus.ABSTAINED, AnswerStatus.BLOCKED}


class EvalTrace(BaseModel):
    """What the system under test reports for one example."""

    answer: str = ""
    status: AnswerStatus | None = None
    citations: list[Citation] = Field(default_factory=list)
    #: First-stage candidate chunk ids (e.g. top-100 before rerank), best first.
    candidate_ids: list[str] = Field(default_factory=list)
    #: Final top-k after fusion / rerank, best first.
    final_ids: list[str] = Field(default_factory=list)
    #: Chunk ids that entered the context window.
    context_ids: list[str] = Field(default_factory=list)
    #: Context texts aligned with ``context_ids`` (needed by faithfulness judges).
    context_texts: list[str] = Field(default_factory=list)
    #: chunk_id -> doc_id; ids absent from the map are treated as doc ids already.
    doc_of: dict[str, str] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    standalone_query: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_docs(self, ids: Sequence[str]) -> list[str]:
        return dedupe(self.doc_of.get(i, i) for i in ids)

    @property
    def total_latency_ms(self) -> float:
        if "total" in self.latency_ms:
            return self.latency_ms["total"]
        return sum(self.latency_ms.values())

    @classmethod
    def from_answer(
        cls,
        answer: Answer,
        *,
        candidates: Sequence[ScoredChunk] = (),
        final: Sequence[ScoredChunk] = (),
        context: Sequence[Chunk] = (),
        cost_usd: float = 0.0,
    ) -> EvalTrace:
        """Build a trace from a pipeline :class:`Answer` plus the stage outputs."""
        doc_of: dict[str, str] = {}
        for sc in [*candidates, *final]:
            doc_of[sc.chunk.chunk_id] = sc.chunk.doc_id
        for ch in context:
            doc_of[ch.chunk_id] = ch.doc_id
        for cit in answer.citations:
            doc_of.setdefault(cit.chunk_id, cit.doc_id)
        context_ids = [c.chunk_id for c in context] or list(answer.context_chunk_ids)
        return cls(
            answer=answer.text,
            status=answer.status,
            citations=list(answer.citations),
            candidate_ids=[s.chunk.chunk_id for s in candidates]
            or list(answer.retrieved_chunk_ids),
            final_ids=[s.chunk.chunk_id for s in final],
            context_ids=context_ids,
            context_texts=[c.text for c in context],
            doc_of=doc_of,
            latency_ms=dict(answer.latency_ms),
            usage=dict(answer.usage),
            cost_usd=cost_usd,
            standalone_query=answer.standalone_query,
        )


SystemUnderTest = Callable[[GoldenExample], Awaitable[EvalTrace]]


class EvalConfig(BaseModel):
    ks: list[int] = Field(default_factory=lambda: list(DEFAULT_KS))
    #: Relevance level for retrieval metrics; ``auto`` = chunk when gold chunks exist.
    level: Literal["auto", "doc", "chunk"] = "doc"
    concurrency: int = Field(default=8, ge=1)
    timeout_s: float = Field(default=120.0, gt=0)
    #: Used to tell F1 (content missing from corpus) apart from F5.
    corpus_doc_ids: set[str] | None = None
    #: Without a correctness judge, an answer "passes" if it contains this share of
    #: reference tokens (lexical fallback; recall is robust to verbose answers).
    lexical_pass_threshold: float = Field(default=0.5, ge=0, le=1)
    #: Store answers / context in results (disable for sensitive corpora).
    store_answers: bool = True


@dataclass
class JudgeSuite:
    """Optional LLM judges; any ``None`` judge is skipped."""

    faithfulness: FaithfulnessJudge | None = None
    correctness: CorrectnessJudge | None = None
    relevance: AnswerRelevanceJudge | None = None
    context: ContextJudge | None = None
    citations: CitationSupportJudge | None = None

    @classmethod
    def all(cls, llm: LLM, config: JudgeConfig | None = None) -> JudgeSuite:
        return cls(
            faithfulness=FaithfulnessJudge(llm, config),
            correctness=CorrectnessJudge(llm, config),
            relevance=AnswerRelevanceJudge(llm, config),
            context=ContextJudge(llm, config),
            citations=CitationSupportJudge(llm, config),
        )

    @classmethod
    def default(cls, llm: LLM, config: JudgeConfig | None = None) -> JudgeSuite:
        """The Phase-0 set: faithfulness + correctness (docs/16-build-plan.md)."""
        return cls(
            faithfulness=FaithfulnessJudge(llm, config), correctness=CorrectnessJudge(llm, config)
        )


class FunnelPosition(BaseModel):
    gold_in_candidates: bool
    gold_in_final: bool
    gold_in_context: bool


class ExampleResult(BaseModel):
    id: str
    question: str
    slices: list[str]
    metrics: dict[str, float] = Field(default_factory=dict)
    abstained: bool = False
    should_abstain: bool = False
    passed: bool | None = None
    funnel: FunnelPosition | None = None
    failure_code: str | None = None
    failure_stage: str | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)
    cost_usd: float = 0.0
    error: str | None = None
    answer: str = ""
    final_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EvalResults(BaseModel):
    name: str
    dataset_name: str
    dataset_version: str
    n: int
    config: dict[str, Any] = Field(default_factory=dict)
    started_at: str
    duration_s: float
    examples: list[ExampleResult]
    aggregate: dict[str, dict[str, float]]
    slice_sizes: dict[str, int]
    funnel: dict[str, float]
    failure_counts: dict[str, int]
    latency: dict[str, dict[str, float]]
    cost: dict[str, float]

    def metric(self, name: str, slice_key: str = "all") -> float | None:
        return self.aggregate.get(slice_key, {}).get(name)

    def per_example(self, name: str) -> dict[str, float]:
        return {e.id: e.metrics[name] for e in self.examples if name in e.metrics}

    def failures(self) -> list[ExampleResult]:
        return [e for e in self.examples if e.passed is False or e.error]

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def load_json(cls, path: str | Path) -> EvalResults:
        """Load a results file, or the ``results`` section of a report JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "results" in data and "examples" not in data:
            data = data["results"]
        return cls.model_validate(data)


def _is_abstained(trace: EvalTrace) -> bool:
    if trace.status in ABSTAIN_STATUSES:
        return True
    if trace.status in (None, AnswerStatus.ANSWERED):
        return is_abstention(trace.answer)
    return False


class EvalRunner:
    def __init__(
        self,
        system: SystemUnderTest,
        config: EvalConfig | None = None,
        judges: JudgeSuite | None = None,
        name: str = "system",
    ) -> None:
        self.system = system
        self.config = config or EvalConfig()
        self.judges = judges or JudgeSuite()
        self.name = name

    # ------------------------------------------------------------------ public

    async def run(self, dataset: GoldenSet) -> EvalResults:
        started = utcnow()
        t0 = time.perf_counter()
        sem = asyncio.Semaphore(self.config.concurrency)

        async def one(ex: GoldenExample) -> ExampleResult:
            async with sem:
                return await self.evaluate_example(ex)

        examples = list(await asyncio.gather(*(one(ex) for ex in dataset)))
        results = self._aggregate(examples, dataset, started.isoformat(), time.perf_counter() - t0)
        logger.info(
            "eval %s on %s (%s): n=%d errors=%d",
            self.name,
            dataset.name,
            results.dataset_version,
            results.n,
            sum(e.error is not None for e in examples),
        )
        return results

    async def evaluate_example(self, ex: GoldenExample) -> ExampleResult:
        res = ExampleResult(
            id=ex.id, question=ex.question, slices=ex.slice_keys(), should_abstain=ex.should_abstain
        )
        try:
            trace = await asyncio.wait_for(self.system(ex), timeout=self.config.timeout_s)
        except TimeoutError:
            res.error = f"timeout after {self.config.timeout_s}s"
            res.passed = False
            return res
        except Exception as exc:  # the system under test may fail arbitrarily
            logger.exception("system under test failed on %s", ex.id)
            res.error = f"{type(exc).__name__}: {exc}"
            res.passed = False
            return res
        try:
            await self._score(ex, trace, res)
        except Exception as exc:  # scoring bugs must not kill the run
            logger.exception("scoring failed on %s", ex.id)
            res.error = f"scoring: {type(exc).__name__}: {exc}"
        return res

    # ------------------------------------------------------------------ scoring

    def _level(self, ex: GoldenExample) -> str:
        if self.config.level == "auto":
            return "chunk" if ex.gold_chunk_ids else "doc"
        return self.config.level

    async def _score(self, ex: GoldenExample, trace: EvalTrace, res: ExampleResult) -> None:
        cfg = self.config
        m = res.metrics
        # Systems may report fewer stages; fall back to the nearest reported one.
        final_ids = trace.final_ids or trace.context_ids
        candidate_ids = trace.candidate_ids or final_ids
        context_ids = trace.context_ids or final_ids
        res.latency_ms = dict(trace.latency_ms)
        res.latency_ms.setdefault("total", trace.total_latency_ms)
        res.cost_usd = trace.cost_usd
        res.final_ids = list(final_ids)
        if cfg.store_answers:
            res.answer = trace.answer
        m["latency_ms"] = res.latency_ms["total"]
        m["cost_usd"] = trace.cost_usd

        abstained = _is_abstained(trace)
        res.abstained = abstained
        m["abstained"] = float(abstained)
        m["abstention_correct"] = float(abstained == ex.should_abstain)

        # ---- retrieval
        has_gold = bool(ex.gold_doc_ids or ex.gold_chunk_ids) and not ex.should_abstain
        if has_gold:
            level = self._level(ex)
            rel = ex.relevance(level)
            if level == "doc":
                final_r, cand_r, ctx_r = (
                    trace.to_docs(final_ids),
                    trace.to_docs(candidate_ids),
                    trace.to_docs(context_ids),
                )
            else:
                final_r, cand_r, ctx_r = (
                    dedupe(final_ids),
                    dedupe(candidate_ids),
                    dedupe(context_ids),
                )
            if rel:
                m.update(retrieval_metrics(final_r, rel, cfg.ks))
                m["candidate_recall"] = recall_at_k(cand_r, rel, max(len(cand_r), 1))
                m["context_hit"] = float(bool(set(rel) & set(ctx_r)))
            gold_docs = ex.gold_doc_ids or [trace.doc_of.get(c, c) for c in ex.gold_chunk_ids]
            cand_docs, final_docs, ctx_docs = (
                trace.to_docs(candidate_ids),
                trace.to_docs(final_ids),
                trace.to_docs(context_ids),
            )
            gs = set(gold_docs)
            res.funnel = FunnelPosition(
                gold_in_candidates=bool(gs & set(cand_docs)),
                gold_in_final=bool(gs & set(final_docs)),
                gold_in_context=bool(gs & set(ctx_docs)),
            )
            code = diagnose(gold_docs, cand_docs, final_docs, ctx_docs, cfg.corpus_doc_ids)
        else:
            code = None

        # ---- lexical answer metrics
        answered = not abstained
        if ex.reference_answer and not ex.should_abstain:
            m["answer_f1"] = token_f1(trace.answer, ex.reference_answer)
            m["exact_match"] = exact_match(trace.answer, ex.reference_answer)
            m["answer_recall"] = token_recall(trace.answer, ex.reference_answer)
        if ex.must_cite and not ex.should_abstain and answered:
            m["citation_present"] = float(bool(trace.citations))
        if trace.citations and ex.gold_doc_ids:
            cgp = cited_gold_precision([c.doc_id for c in trace.citations], ex.gold_doc_ids)
            if cgp is not None:
                m["cited_gold_precision"] = cgp

        # ---- LLM judges (run concurrently)
        await self._judge(ex, trace, res, answered)

        # ---- pass/fail + failure attribution
        if ex.should_abstain:
            res.passed = abstained
            if not abstained:
                res.failure_stage = "abstention"
                res.notes.append("answered an unanswerable question")
        else:
            if abstained:
                res.passed = False
            elif "correct" in m:
                res.passed = bool(m["correct"])
            elif "answer_recall" in m:
                res.passed = m["answer_recall"] >= cfg.lexical_pass_threshold
            if res.passed is not None and "faithfulness" in m and m["faithfulness"] < 1.0:
                res.notes.append(f"faithfulness {m['faithfulness']:.2f}")
            if res.passed is False:
                if code is not None:
                    res.failure_code = code.value
                    res.failure_stage = FAILURE_STAGE[code]
                else:
                    res.failure_stage = "generation"
                    if abstained and res.funnel is not None and res.funnel.gold_in_context:
                        res.failure_code = FailureCode.F8_NOT_EXTRACTED.value
                        res.notes.append("abstained although gold evidence was in context")
        if res.passed is not None:
            m["pass"] = float(res.passed)

    async def _judge(
        self, ex: GoldenExample, trace: EvalTrace, res: ExampleResult, answered: bool
    ) -> None:
        j = self.judges
        m = res.metrics
        tasks: dict[str, Awaitable[Any]] = {}
        contexts = list(trace.context_texts)
        if j.faithfulness and answered and (contexts or not trace.context_ids):
            tasks["faithfulness"] = j.faithfulness.judge(ex.question, trace.answer, contexts)
        if j.correctness and answered and ex.reference_answer and not ex.should_abstain:
            tasks["correctness"] = j.correctness.judge(
                ex.question, trace.answer, ex.reference_answer
            )
        if j.relevance and answered:
            tasks["relevance"] = j.relevance.judge(ex.question, trace.answer)
        if j.context and contexts:
            tasks["context_relevance"] = j.context.relevance(ex.question, contexts)
            if ex.reference_answer and not ex.should_abstain:
                tasks["context_recall"] = j.context.recall(
                    ex.question, ex.reference_answer, contexts
                )
        if j.citations and answered and trace.citations:
            text_of = dict(zip(trace.context_ids, trace.context_texts, strict=False))
            sources = {
                c.index: c.snippet or text_of.get(c.chunk_id, "")
                for c in trace.citations
                if c.snippet or c.chunk_id in text_of
            }
            if sources:
                tasks["citations"] = j.citations.judge(trace.answer, sources)
        if not tasks:
            return
        outs = dict(zip(tasks, await asyncio.gather(*tasks.values()), strict=True))
        f = outs.get("faithfulness")
        if f is not None and f.score is not None:
            m["faithfulness"] = f.score
            unsupported = f.unsupported
            if unsupported:
                res.notes.append("unsupported claims: " + "; ".join(unsupported[:3]))
        c = outs.get("correctness")
        if c is not None:
            m["correct"] = float(c.correct)
            m["correctness_score"] = float(c.score)
        r = outs.get("relevance")
        if r is not None:
            m["answer_relevant"] = float(r.relevant)
            m["answer_relevance_score"] = float(r.score)
        for key in ("context_relevance", "context_recall"):
            if outs.get(key) is not None:
                m[key] = float(outs[key])
        cs = outs.get("citations")
        if cs:
            scores = citation_scores(cs)
            m["citation_precision"] = scores.citation_precision
            m["citation_recall"] = scores.citation_recall

    # ------------------------------------------------------------------ aggregation

    def _aggregate(
        self, examples: list[ExampleResult], dataset: GoldenSet, started: str, duration: float
    ) -> EvalResults:
        sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        sizes: Counter[str] = Counter()
        for e in examples:
            for s in e.slices:
                sizes[s] += 1
                for k, v in e.metrics.items():
                    if not math.isnan(v):
                        sums[s][k].append(v)
        aggregate: dict[str, dict[str, float]] = {
            s: {k: sum(v) / len(v) for k, v in sorted(ms.items()) if v} for s, ms in sums.items()
        }
        n = len(examples)
        errors = sum(e.error is not None for e in examples)
        allm = aggregate.setdefault("all", {})
        allm["error_rate"] = errors / n if n else 0.0
        scored = [e for e in examples if e.error is None]
        allm.update(abstention_report((e.should_abstain, e.abstained) for e in scored).as_metrics())

        # Latency per stage and cost.
        stages: dict[str, list[float]] = defaultdict(list)
        for e in scored:
            for stage, ms in e.latency_ms.items():
                stages[stage].append(ms)
        latency = {
            stage: {
                "p50": percentile(v, 50),
                "p95": percentile(v, 95),
                "mean": sum(v) / len(v),
            }
            for stage, v in sorted(stages.items())
            if v
        }
        if "total" in latency:
            allm["latency_p50_ms"] = latency["total"]["p50"]
            allm["latency_p95_ms"] = latency["total"]["p95"]
        costs = [e.cost_usd for e in scored]
        cost = {
            "total_usd": sum(costs),
            "per_query_usd": sum(costs) / len(costs) if costs else 0.0,
        }
        allm["cost_per_query_usd"] = cost["per_query_usd"]

        # Retrieval funnel over answerable examples with gold.
        with_gold = [e for e in scored if e.funnel is not None]
        funnel: dict[str, float] = {"n": float(len(with_gold))}
        if with_gold:
            g = len(with_gold)
            funnel["gold_in_candidates"] = (
                sum(e.funnel.gold_in_candidates for e in with_gold if e.funnel) / g
            )
            funnel["gold_in_final"] = sum(e.funnel.gold_in_final for e in with_gold if e.funnel) / g
            funnel["gold_in_context"] = (
                sum(e.funnel.gold_in_context for e in with_gold if e.funnel) / g
            )
            passed = [e for e in with_gold if e.passed]
            funnel["answered_correctly"] = len(passed) / g
        failure_counts = Counter(
            e.failure_code or f"stage:{e.failure_stage}"
            for e in examples
            if e.passed is False and not e.error
        )
        if errors:
            failure_counts["error"] = errors
        return EvalResults(
            name=self.name,
            dataset_name=dataset.name,
            dataset_version=dataset.version_hash(),
            n=n,
            config=self.config.model_dump(mode="json", exclude={"corpus_doc_ids"}),
            started_at=started,
            duration_s=duration,
            examples=examples,
            aggregate=aggregate,
            slice_sizes=dict(sorted(sizes.items())),
            funnel=funnel,
            failure_counts=dict(failure_counts.most_common()),
            latency=latency,
            cost=cost,
        )
