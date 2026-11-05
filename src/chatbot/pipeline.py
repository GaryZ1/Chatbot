"""Online / serving plane (docs/00-architecture-overview.md §2, steps a-m):

authN/Z -> input guard -> condense / understand / route -> hybrid retrieval with the
ACL pre-filter -> indirect-injection scan -> rerank -> abstention gate -> small-to-big
expansion -> context assembly -> generation -> output guard (groundedness, citations,
sanitisation) -> memory, cache, trace, metrics.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel

from chatbot.build import Components
from chatbot.core.filters import Condition, Filter, Op
from chatbot.core.types import Answer, AnswerStatus, Principal, ScoredChunk
from chatbot.generate import StreamEvent
from chatbot.guard.verdicts import GuardAction, OutputAction
from chatbot.llm import LLMError, LLMRefusal
from chatbot.memory.session import Session
from chatbot.query import QueryPlan, Route
from chatbot.rerank import postprocess_context
from chatbot.retrieve import apply_boosts, expand_neighbours, fuse_many, resolve_parents
from chatbot.retrieve.mmr import MMRSelector

log = logging.getLogger(__name__)

ESCALATE_TEXT = (
    "I'll hand this over to a member of the team, who can help with this request. "
    "A summary of our conversation has been attached."
)
CLARIFY_FALLBACK = "Could you tell me a bit more about what you are looking for?"
REGENERATE_NOTE = (
    "\n\n(Your previous draft contained claims that were not supported by the sources. "
    "Answer again using ONLY the sources and cite every claim.)"
)


class RetrievalResult(BaseModel):
    candidates: list[ScoredChunk]
    reranked: list[ScoredChunk]
    context: list[ScoredChunk]
    quarantined: list[str] = []
    abstain: bool = False


class RAGPipeline:
    def __init__(self, components: Components) -> None:
        self.c = components
        self._mmr = (
            MMRSelector(components.embedder, lambda_=components.settings.retrieval.mmr_lambda)
            if components.settings.retrieval.mmr_lambda is not None
            else None
        )

    # ------------------------------------------------------------------ retrieval

    def _base_filter(self, plan: QueryPlan | None, extra: Filter | None) -> Filter | None:
        flt = Filter()
        if self.c.settings.retrieval.only_latest:
            flt = flt.and_(Filter(must_not=[Condition(field="is_latest", op=Op.EQ, value=False)]))
        if plan is not None and plan.filter is not None:
            flt = flt.and_(plan.filter)
        flt = flt.and_(extra)
        return None if flt.is_empty() else flt

    async def retrieve(
        self,
        queries: Sequence[str],
        principal: Principal,
        flt: Filter | None = None,
        *,
        rerank_query: str | None = None,
    ) -> RetrievalResult:
        """Hybrid retrieval for one or more queries (multi-query / sub-questions), fused
        with RRF, then rerank -> gate -> expansion -> post-processing."""
        r = self.c.settings.retrieval
        queries = [q for q in dict.fromkeys(queries) if q.strip()]
        if not queries:
            return RetrievalResult(candidates=[], reranked=[], context=[], abstain=True)
        lists = [await self.c.retriever.retrieve(q, r.candidates, principal, flt) for q in queries]
        candidates = lists[0] if len(lists) == 1 else fuse_many(lists, k=60, top_k=r.candidates)

        quarantined: list[str] = []
        if self.c.chunk_scanner is not None and candidates:
            scan = await self.c.chunk_scanner.scan_chunks(candidates)
            if scan.quarantined:
                quarantined = scan.quarantined_ids
                log.warning("quarantined %d retrieved chunk(s): %s", len(quarantined), quarantined)
                drop = set(quarantined)
                candidates = [h for h in candidates if h.chunk.chunk_id not in drop]

        q = rerank_query or queries[0]
        if self.c.reranker is not None and candidates:
            reranked = await self.c.reranker.rerank(
                q, candidates, top_k=min(len(candidates), 3 * r.final_k)
            )
        else:
            reranked = list(candidates)

        abstain = False
        if self.c.gate is not None and reranked:
            decision = self.c.gate.decide(reranked)
            abstain = decision.abstain
            reranked = self.c.gate.filter(reranked) or reranked
        if not reranked:
            abstain = True

        if r.freshness_weight:
            reranked = apply_boosts(reranked, freshness_weight=r.freshness_weight)
        if self._mmr is not None:
            selected = await self._mmr.select(q, reranked, r.final_k)
        else:
            selected = reranked[: r.final_k]
        if r.neighbour_window:
            selected = await expand_neighbours(
                selected, self.c.docstore, principal, flt, window=r.neighbour_window
            )
        if r.parent_expansion:
            selected = await resolve_parents(selected, self.c.docstore, principal, flt)
        context = postprocess_context(selected, order="score")
        return RetrievalResult(
            candidates=candidates,
            reranked=reranked,
            context=context,
            quarantined=quarantined,
            abstain=abstain,
        )

    # ------------------------------------------------------------------ chat

    async def _blocked(self, text: str, flags: Sequence[str], status: AnswerStatus) -> Answer:
        return Answer(text=text, status=status, guardrail_flags=list(flags), route="GUARD")

    async def _queries_for(self, plan: QueryPlan) -> list[str]:
        r = self.c.settings.retrieval
        queries = list(plan.retrieval_queries)
        if r.multi_query and plan.route is Route.RAG_SIMPLE:
            from chatbot.query import MultiQueryGenerator

            try:
                queries = await MultiQueryGenerator(self.c.llm, n=r.multi_query).generate(
                    plan.standalone_query
                )
            except LLMError:
                log.warning("multi-query failed; using the standalone query only")
        if r.hyde and plan.route is Route.RAG_SIMPLE:
            from chatbot.query import HyDEGenerator

            try:
                hy = await HyDEGenerator(self.c.llm).generate(plan.standalone_query)
                if hy.applied:
                    queries.append(hy.embed_text)
            except LLMError:
                log.warning("HyDE failed; skipping")
        return queries

    async def chat(
        self,
        message: str,
        principal: Principal,
        *,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        capture: dict[str, Any] | None = None,
    ) -> tuple[Answer, Session]:
        """Answer one user turn. Never raises for model or guard outcomes; those become
        an :class:`Answer` with the corresponding status."""
        events = [
            e
            async for e in self.chat_stream(
                message, principal, session_id=session_id, filters=filters, capture=capture
            )
        ]
        final = events[-1]
        assert final.answer is not None and final.session is not None
        return final.answer, final.session

    async def chat_stream(
        self,
        message: str,
        principal: Principal,
        *,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        stream: bool = False,
        capture: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Like :meth:`chat` but yields ``delta`` events while generating (``stream=True``).

        ``capture`` (optional, for evaluation) receives the stage-level retrieval result.
        """
        c = self.c
        t_start = time.perf_counter()
        session = await c.memory.load(session_id, principal)
        async with c.tracer.trace(query=message, principal=principal) as t:
            answer: Answer | None = None
            text = message

            # b. input guardrails
            if c.input_guard is not None:
                with t.span("input_guard"):
                    verdict = await c.input_guard.check(message, principal)
                t.add_verdict(verdict)
                if verdict.action in (GuardAction.BLOCK, GuardAction.ESCALATE):
                    status = (
                        AnswerStatus.ESCALATED
                        if verdict.action is GuardAction.ESCALATE
                        else AnswerStatus.BLOCKED
                    )
                    text_out = (
                        "Too many requests, please retry shortly."
                        if {"rate_limited", "budget_exceeded"} & set(verdict.flags)
                        else "I can't help with that request."
                    )
                    answer = await self._blocked(text_out, verdict.flags, status)
                text = verdict.text_or(message)

            # semantic cache: only first turns (follow-ups depend on history)
            first_turn = not session.turns and not session.summary
            if answer is None and c.cache is not None and first_turn and not filters:
                hit = await c.cache.get(text, principal, c.index_version)
                if hit is not None:
                    answer = hit.answer.model_copy(update={"trace_id": t.trace_id})
                    answer.guardrail_flags = [*answer.guardrail_flags, f"cache_{hit.kind}"]

            if answer is None:
                async for ev in self._answer(
                    text, principal, session, filters, t, stream=stream, capture=capture
                ):
                    if ev.type == "delta":
                        yield ev
                    else:
                        answer = ev.answer
                assert answer is not None

            answer.trace_id = t.trace_id
            answer.latency_ms["total"] = (time.perf_counter() - t_start) * 1000
            t.set_answer(answer)
            t.set(route=answer.route, standalone_query=answer.standalone_query)

        # m. memory + accounting (outside the trace span, never fails the request)
        try:
            await c.memory.add_exchange(session, message, answer.text)
        except Exception:
            log.exception("failed to update session memory")
        tokens = sum(answer.usage.values()) if answer.usage else 0
        c.limits.record(principal, tokens=tokens)
        c.metrics.observe_request(
            route=answer.route or "unknown",
            status=answer.status.value,
            latency_ms=answer.latency_ms,
            cache_hit=any(f.startswith("cache_") for f in answer.guardrail_flags),
        )
        yield ChatEvent(type="answer", answer=answer, session=session)

    async def _answer(
        self,
        text: str,
        principal: Principal,
        session: Session,
        filters: dict[str, Any] | None,
        t: Any,
        *,
        stream: bool,
        capture: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        c = self.c
        gen_ctx = c.memory.context_for_generation(session)

        # c/d. condense + understand + route (one structured small-model call)
        with t.span("understand"):
            plan = await c.understanding.understand(
                text, c.memory.history_for_condensing(session), scope_key=principal.scope_key()
            )
        t.set(route=plan.route.value, condensed_query=plan.standalone_query, filters=plan.filters)

        if plan.route is Route.SMALLTALK:
            try:
                ans = await c.generator.smalltalk(text, recent_turns=gen_ctx.recent_turns)
            except LLMError:
                ans = Answer(text="Hello! How can I help?", status=AnswerStatus.SMALLTALK)
            ans.route = plan.route.value
            yield ChatEvent(type="answer", answer=ans)
            return
        if plan.route is Route.CLARIFY:
            yield ChatEvent(
                type="answer",
                answer=Answer(
                    text=plan.clarify_question or CLARIFY_FALLBACK,
                    status=AnswerStatus.CLARIFY,
                    route=plan.route.value,
                    standalone_query=plan.standalone_query,
                ),
            )
            return
        if plan.route in (Route.ESCALATE, Route.SQL_TOOL):
            yield ChatEvent(
                type="answer",
                answer=Answer(
                    text=ESCALATE_TEXT, status=AnswerStatus.ESCALATED, route=plan.route.value
                ),
            )
            return
        if plan.route is Route.AGENT and c.settings.agent.enabled:
            from chatbot.agent import AgentBudget, AgenticRAG

            with t.span("agent"):
                result = await AgenticRAG(
                    c.llm,
                    c.retriever,
                    c.docstore,
                    budget=AgentBudget(max_steps=c.settings.agent.max_steps),
                    base_filter=self._base_filter(None, None),
                ).run(plan.standalone_query, principal, gen_ctx.recent_turns)
            ans = result.answer
            ans.route = plan.route.value
            yield ChatEvent(type="answer", answer=ans)
            return

        # e-g. retrieval
        extra = None
        if filters:
            extra = Filter(
                must=[
                    Condition(field=k, op=Op.IN if isinstance(v, list) else Op.EQ, value=v)
                    for k, v in filters.items()
                ]
            )
        flt = self._base_filter(plan, extra)
        with t.span("retrieve"):
            queries = await self._queries_for(plan)
            res = await self.retrieve(queries, principal, flt, rerank_query=plan.standalone_query)
        if capture is not None:
            capture["retrieval"] = res
            capture["plan"] = plan
        t.record_retrieval("candidates", res.candidates)
        t.record_retrieval("reranked", res.reranked)
        t.record_context(res.context)

        evidence = [] if res.abstain else res.context
        summary = gen_ctx.summary_block()
        memories = ""
        if c.long_term is not None:
            items = await c.long_term.retrieve(principal, plan.standalone_query)
            memories = c.long_term.format_for_prompt(items)

        # h-i. generation (streamed or not)
        answer: Answer | None = None
        with t.span("generate"):
            try:
                if stream and evidence:
                    async for ev in c.generator.stream(
                        text,
                        evidence,
                        summary=summary,
                        recent_turns=gen_ctx.recent_turns,
                        memories=memories,
                        standalone_query=plan.standalone_query,
                        route=plan.route.value,
                    ):
                        if ev.type == "delta":
                            yield ChatEvent(type="delta", text=ev.text)
                        else:
                            answer = ev.answer
                else:
                    answer = await c.generator.generate(
                        text,
                        evidence,
                        summary=summary,
                        recent_turns=gen_ctx.recent_turns,
                        memories=memories,
                        standalone_query=plan.standalone_query,
                        route=plan.route.value,
                    )
            except LLMRefusal:
                answer = Answer(
                    text="I can't help with that request.",
                    status=AnswerStatus.BLOCKED,
                    route=plan.route.value,
                )
            except LLMError:
                log.exception("generation failed")
                answer = Answer(
                    text="Sorry, I'm having trouble answering right now. Please try again shortly.",
                    status=AnswerStatus.ESCALATED,
                    route=plan.route.value,
                    guardrail_flags=["llm_error"],
                )
        assert answer is not None
        answer.retrieved_chunk_ids = [h.chunk.chunk_id for h in res.candidates]
        if res.quarantined:
            answer.guardrail_flags.append("quarantined_chunks")

        # j. output guardrails (+ one regeneration)
        if c.output_guard is not None and answer.status in (
            AnswerStatus.ANSWERED,
            AnswerStatus.PARTIAL,
        ):
            pack = c.generator.prepare(evidence)
            sources = [pack.sources[i] for i in sorted(pack.sources)]
            with t.span("output_guard"):
                ov = await c.output_guard.check(answer.text, sources)
                if ov.action is OutputAction.REGENERATE and not stream:
                    try:
                        answer = await c.generator.generate(
                            text + REGENERATE_NOTE,
                            pack,
                            summary=summary,
                            recent_turns=gen_ctx.recent_turns,
                            standalone_query=plan.standalone_query,
                            route=plan.route.value,
                        )
                        ov = await c.output_guard.check(answer.text, sources, attempt=1)
                    except LLMError:
                        log.exception("regeneration failed")
                    if ov.action is OutputAction.REGENERATE:
                        ov.action = OutputAction.FALLBACK
            t.add_verdict(ov)
            answer.guardrail_flags = [*answer.guardrail_flags, *ov.flags]
            if ov.action is OutputAction.DELIVER or (
                stream and ov.action is OutputAction.REGENERATE
            ):
                answer.text = ov.text
            elif ov.action is OutputAction.ESCALATE:
                answer.text, answer.status = ov.text, AnswerStatus.ESCALATED
            else:
                answer.text = c.output_guard.policy.fallback_text
                answer.status = AnswerStatus.ABSTAINED
                answer.citations = []

        if (
            c.cache is not None
            and not session.turns
            and not session.summary
            and not filters
            and answer.status is AnswerStatus.ANSWERED
        ):
            await c.cache.put(
                text,
                principal,
                c.index_version,
                answer,
                doc_ids=sorted({ci.doc_id for ci in answer.citations}),
            )
        yield ChatEvent(type="answer", answer=answer)


class ChatEvent(BaseModel):
    """Streaming event: text deltas, then exactly one final ``answer`` event."""

    type: str
    text: str = ""
    answer: Answer | None = None
    session: Session | None = None

    def as_stream_event(self) -> StreamEvent:
        return StreamEvent(
            type="delta" if self.type == "delta" else "answer", text=self.text, answer=self.answer
        )
