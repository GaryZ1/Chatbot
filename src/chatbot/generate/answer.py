"""Grounded answer generation (docs/10-generation-context.md §2-§5).

:class:`AnswerGenerator` assembles the request in the §2 layout (stable system prompt,
then sources, conversation summary, recent turns and the question repeated last),
calls the LLM (mid tier by default; route-based tiering per §4), and post-processes
the output into a :class:`~chatbot.core.types.Answer`: citations parsed and mapped,
out-of-range markers removed, abstention / partial status detected, and optional
claim-level citation verification.

Sampling parameters (temperature, top_p) are intentionally never sent: current
models fix them, and factual QA relies on the grounding policy instead.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator, Sequence
from typing import Literal

from pydantic import BaseModel

from chatbot.core.types import Answer, AnswerStatus, ScoredChunk, Turn
from chatbot.generate.citations import CitationVerifier, extract_citations
from chatbot.generate.context_builder import ContextBuilder, ContextPack
from chatbot.generate.prompts import (
    ABSTAIN_PHRASE,
    PARTIAL_HEADING,
    PROMPT_VERSION,
    SMALLTALK_SYSTEM,
    PromptConfig,
    build_system_prompt,
    render_turns,
    render_user_message,
)
from chatbot.llm.base import LLM, LLMMessage, LLMRefusal, LLMRequest, LLMResponse, Tier, Usage

logger = logging.getLogger(__name__)

_ABSTAIN_PATTERNS = [
    re.compile(re.escape(ABSTAIN_PHRASE.rstrip(".")), re.IGNORECASE),
    re.compile(r"\bI (?:could not|couldn't|can't|cannot|was unable to|am unable to) find\b", re.I),
    re.compile(
        r"\b(?:the )?(?:provided |available |given )?(?:sources|documents|context) (?:do not|don't|does not) "
        r"(?:contain|mention|include|cover|provide|say)\b",
        re.I,
    ),
    re.compile(r"\bno (?:relevant )?information (?:is |was )?(?:available|found|provided)\b", re.I),
    re.compile(r"\bI don't (?:know|have (?:enough )?information)\b", re.I),
]
_PARTIAL_RE = re.compile(re.escape(PARTIAL_HEADING.rstrip(":")), re.IGNORECASE)

DEFAULT_NO_CONTEXT_MESSAGE = (
    f"{ABSTAIN_PHRASE} Could you rephrase the question or add details (product, version, date)? "
    "I can also connect you with a human if you prefer."
)
REFUSAL_MESSAGE = "I can't help with that request."


def detect_abstention(text: str, n_citations: int) -> bool:
    """Abstained = an abstention phrase in the opening and no valid citations."""
    head = text.strip()[:400]
    return n_citations == 0 and any(p.search(head) for p in _ABSTAIN_PATTERNS)


def detect_partial(text: str, n_citations: int) -> bool:
    if n_citations == 0:
        return False
    return bool(_PARTIAL_RE.search(text)) or any(p.search(text) for p in _ABSTAIN_PATTERNS)


def _usage_dict(u: Usage) -> dict[str, int]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_input_tokens": u.cache_read_input_tokens,
        "cache_creation_input_tokens": u.cache_creation_input_tokens,
    }


class StreamEvent(BaseModel):
    type: Literal["delta", "answer"]
    text: str = ""
    answer: Answer | None = None


Evidence = Sequence[ScoredChunk] | ContextPack


class AnswerGenerator:
    """Generates cited answers from retrieved evidence.

    Args:
        llm: client (use a :class:`~chatbot.llm.gateway.FailoverLLM` in production).
        context_builder: budgeted ``<sources>`` assembly.
        prompt_config: system-prompt options; the prompt is built once (cache-stable).
        tier: model tier for answers (mid by default; frontier for hard synthesis).
        max_tokens: output cap (thinking tokens count against it on current models).
        verifier: when set, citations are verified per claim and ``supported`` is filled.
        abstain_on_empty_context: skip the LLM and abstain when no sources survive.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        context_builder: ContextBuilder | None = None,
        prompt_config: PromptConfig | None = None,
        tier: Tier = Tier.MID,
        max_tokens: int = 8000,
        effort: str | None = None,
        verifier: CitationVerifier | None = None,
        abstain_on_empty_context: bool = True,
        no_context_message: str = DEFAULT_NO_CONTEXT_MESSAGE,
    ) -> None:
        self.llm = llm
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_config = prompt_config or PromptConfig()
        self.system_prompt = build_system_prompt(self.prompt_config)
        self.tier = tier
        self.max_tokens = max_tokens
        self.effort = effort
        self.verifier = verifier
        self.abstain_on_empty_context = abstain_on_empty_context
        self.no_context_message = no_context_message

    # ------------------------------------------------------------------ request

    def prepare(self, evidence: Evidence) -> ContextPack:
        if isinstance(evidence, ContextPack):
            return evidence
        return self.context_builder.build(list(evidence))

    def build_request(
        self,
        question: str,
        pack: ContextPack,
        *,
        summary: str = "",
        recent_turns: Sequence[Turn] = (),
        memories: str = "",
        standalone_query: str | None = None,
        tier: Tier | None = None,
    ) -> LLMRequest:
        if not question.strip():
            raise ValueError("question must be non-empty")
        user = render_user_message(
            question,
            pack.xml,
            summary=summary,
            recent_turns=recent_turns,
            memories=memories,
            standalone_query=standalone_query,
        )
        return LLMRequest(
            messages=[LLMMessage(role="user", content=user)],
            system=self.system_prompt,
            tier=tier or self.tier,
            max_tokens=self.max_tokens,
            effort=self.effort,
            cache_system=True,
            metadata={"prompt_version": PROMPT_VERSION, "stage": "generate"},
        )

    # ------------------------------------------------------------------ post-processing

    async def finalize(
        self,
        raw_text: str,
        pack: ContextPack,
        *,
        route: str = "",
        standalone_query: str = "",
        usage: Usage | None = None,
        latency_ms: float = 0.0,
        stop_reason: str = "end_turn",
    ) -> Answer:
        """Turn raw model text into an :class:`Answer` (citations, status, flags)."""
        result = extract_citations(raw_text, pack.sources)
        flags: list[str] = []
        if result.invalid_indices:
            flags.append("invalid_citation")
        n = len(result.citations)
        if detect_abstention(result.text, n):
            status = AnswerStatus.ABSTAINED
        elif detect_partial(result.text, n) or stop_reason == "max_tokens":
            status = AnswerStatus.PARTIAL
        else:
            status = AnswerStatus.ANSWERED
        if stop_reason == "max_tokens":
            flags.append("truncated")
        if status is AnswerStatus.ANSWERED and n == 0 and not pack.is_empty:
            flags.append("uncited_answer")
        citations = result.citations
        if self.verifier is not None and citations:
            report = await self.verifier.verify(result.text, pack.sources, citations)
            citations = report.citations
            if report.unsupported_claims:
                flags.append("unsupported_claims")
        return Answer(
            text=result.text,
            status=status,
            citations=citations,
            route=route,
            standalone_query=standalone_query,
            context_chunk_ids=pack.chunk_ids,
            guardrail_flags=flags,
            usage=_usage_dict(usage or Usage()),
            latency_ms={"generate": latency_ms},
        )

    def _no_context_answer(self, route: str, standalone_query: str) -> Answer:
        return Answer(
            text=self.no_context_message,
            status=AnswerStatus.ABSTAINED,
            route=route,
            standalone_query=standalone_query,
            guardrail_flags=["no_context"],
        )

    # ------------------------------------------------------------------ generation

    async def generate(
        self,
        question: str,
        evidence: Evidence,
        *,
        summary: str = "",
        recent_turns: Sequence[Turn] = (),
        memories: str = "",
        standalone_query: str | None = None,
        route: str = "",
        tier: Tier | None = None,
    ) -> Answer:
        pack = self.prepare(evidence)
        sq = standalone_query or ""
        if pack.is_empty and self.abstain_on_empty_context:
            return self._no_context_answer(route, sq)
        req = self.build_request(
            question,
            pack,
            summary=summary,
            recent_turns=recent_turns,
            memories=memories,
            standalone_query=standalone_query,
            tier=tier,
        )
        started = time.perf_counter()
        try:
            resp: LLMResponse = await self.llm.generate(req)
        except LLMRefusal as exc:
            logger.warning("generation refused: %s", exc)
            return Answer(
                text=REFUSAL_MESSAGE,
                status=AnswerStatus.BLOCKED,
                route=route,
                standalone_query=sq,
                context_chunk_ids=pack.chunk_ids,
                guardrail_flags=["model_refusal"],
            )
        latency = (time.perf_counter() - started) * 1000.0
        return await self.finalize(
            resp.text,
            pack,
            route=route,
            standalone_query=sq,
            usage=resp.usage,
            latency_ms=latency,
            stop_reason=resp.stop_reason,
        )

    async def stream(
        self,
        question: str,
        evidence: Evidence,
        *,
        summary: str = "",
        recent_turns: Sequence[Turn] = (),
        memories: str = "",
        standalone_query: str | None = None,
        route: str = "",
        tier: Tier | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield ``delta`` events as text arrives, then one final ``answer`` event.

        Deltas are raw model text (markers included); the final answer carries the
        cleaned text, validated citations and status. Streaming APIs do not report
        usage here; meter tokens at the gateway.
        """
        pack = self.prepare(evidence)
        sq = standalone_query or ""
        if pack.is_empty and self.abstain_on_empty_context:
            ans = self._no_context_answer(route, sq)
            yield StreamEvent(type="delta", text=ans.text)
            yield StreamEvent(type="answer", answer=ans)
            return
        req = self.build_request(
            question,
            pack,
            summary=summary,
            recent_turns=recent_turns,
            memories=memories,
            standalone_query=standalone_query,
            tier=tier,
        )
        started = time.perf_counter()
        first_token_ms: float | None = None
        parts: list[str] = []
        try:
            async for delta in self.llm.stream(req):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000.0
                parts.append(delta)
                yield StreamEvent(type="delta", text=delta)
        except LLMRefusal as exc:
            logger.warning("streamed generation refused: %s", exc)
            yield StreamEvent(
                type="answer",
                answer=Answer(
                    text=REFUSAL_MESSAGE,
                    status=AnswerStatus.BLOCKED,
                    route=route,
                    standalone_query=sq,
                    guardrail_flags=["model_refusal"],
                ),
            )
            return
        latency = (time.perf_counter() - started) * 1000.0
        answer = await self.finalize("".join(parts), pack, route=route, standalone_query=sq, latency_ms=latency)
        if first_token_ms is not None:
            answer.latency_ms["first_token"] = first_token_ms
        yield StreamEvent(type="answer", answer=answer)

    async def smalltalk(
        self, message: str, *, recent_turns: Sequence[Turn] = (), route: str = "SMALLTALK"
    ) -> Answer:
        """Reply to greetings / out-of-scope messages without retrieval (small tier)."""
        prompt = message
        if recent_turns:
            prompt = f"<recent_turns>\n{render_turns(recent_turns[-4:])}\n</recent_turns>\n\n{message}"
        req = LLMRequest.simple(prompt, system=SMALLTALK_SYSTEM, tier=Tier.SMALL, max_tokens=1024)
        resp = await self.llm.generate(req)
        return Answer(
            text=resp.text.strip(),
            status=AnswerStatus.SMALLTALK,
            route=route,
            usage=_usage_dict(resp.usage),
        )
