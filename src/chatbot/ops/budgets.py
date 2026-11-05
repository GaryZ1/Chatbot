"""Request-level budgets: LLM-call, token, agent-step caps and deadlines.

Implements the "agent step caps" and denial-of-wallet controls of
docs/13-guardrails-security.md §1 and the per-route latency budget of
docs/14-production-llmops.md §1 (agentic routes get their own, looser SLO).
A :class:`RequestBudget` is created per request and charged by every LLM call /
agent step; exceeding a cap raises :class:`~chatbot.ops.ratelimit.BudgetExceeded`
(``kind`` says which), which the pipeline turns into a partial answer or fallback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

from pydantic import BaseModel, Field

from chatbot.llm.base import Usage
from chatbot.ops.ratelimit import BudgetExceeded

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DeadlineExceeded(BudgetExceeded, TimeoutError):
    """The request (or a stage) ran past its deadline."""

    def __init__(self, message: str, stage: str = "") -> None:
        super().__init__(message, 0.0, "", kind="deadline")
        self.stage = stage


class BudgetLimits(BaseModel):
    """Caps for one request. Defaults suit the standard RAG route; agentic routes raise them."""

    max_llm_calls: int = Field(default=8, ge=1)
    max_tokens: int = Field(default=60_000, ge=1)  # input + output across calls
    max_agent_steps: int = Field(default=6, ge=1)
    max_tool_calls: int = Field(default=10, ge=0)
    max_usd: float | None = Field(default=None, gt=0)
    deadline_s: float = Field(default=30.0, gt=0)

    @classmethod
    def agentic(cls) -> BudgetLimits:
        return cls(
            max_llm_calls=20,
            max_tokens=250_000,
            max_agent_steps=12,
            max_tool_calls=30,
            deadline_s=120.0,
        )


class RequestBudget:
    """Mutable per-request budget tracker."""

    def __init__(
        self, limits: BudgetLimits | None = None, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limits = limits or BudgetLimits()
        self._clock = clock
        self.started = clock()
        self.llm_calls = 0
        self.tokens = 0
        self.agent_steps = 0
        self.tool_calls = 0
        self.usd = 0.0

    # ------------------------------------------------------------------ charging

    def _exceed(self, kind: str, msg: str) -> None:
        logger.warning("request budget exceeded (%s): %s", kind, msg)
        raise BudgetExceeded(msg, 0.0, "", kind=kind)

    def check_llm_call(self) -> None:
        """Call *before* an LLM request: raises if another call is not allowed."""
        self.check_deadline()
        if self.llm_calls >= self.limits.max_llm_calls:
            self._exceed("llm_calls", f"max {self.limits.max_llm_calls} LLM calls per request")
        if self.tokens >= self.limits.max_tokens:
            self._exceed("tokens", f"max {self.limits.max_tokens} tokens per request")

    def charge_llm(self, usage: Usage, *, usd: float = 0.0) -> None:
        """Record a finished LLM call; raises if it pushed the request over a cap."""
        self.llm_calls += 1
        self.tokens += (
            usage.input_tokens
            + usage.output_tokens
            + usage.cache_read_input_tokens
            + usage.cache_creation_input_tokens
        )
        self.usd += usd
        if self.tokens > self.limits.max_tokens:
            self._exceed("tokens", f"used {self.tokens} > {self.limits.max_tokens} tokens")
        if self.limits.max_usd is not None and self.usd > self.limits.max_usd:
            self._exceed("usd", f"spent ${self.usd:.4f} > ${self.limits.max_usd:.4f}")

    def step(self) -> int:
        """Count one agent step; returns the step number (1-based)."""
        self.check_deadline()
        if self.agent_steps >= self.limits.max_agent_steps:
            self._exceed("agent_steps", f"max {self.limits.max_agent_steps} agent steps")
        self.agent_steps += 1
        return self.agent_steps

    def tool_call(self) -> None:
        if self.tool_calls >= self.limits.max_tool_calls:
            self._exceed("tool_calls", f"max {self.limits.max_tool_calls} tool calls")
        self.tool_calls += 1

    # ------------------------------------------------------------------ time

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self.started

    def remaining_s(self) -> float:
        return max(0.0, self.limits.deadline_s - self.elapsed_s)

    def check_deadline(self) -> None:
        if self.remaining_s() <= 0:
            raise DeadlineExceeded(f"request deadline of {self.limits.deadline_s}s exceeded")

    @asynccontextmanager
    async def deadline(
        self, stage: str = "", *, stage_timeout_s: float | None = None
    ) -> AsyncIterator[None]:
        """``async with budget.deadline("rerank", stage_timeout_s=0.3):`` bounded by the request deadline."""
        self.check_deadline()
        timeout = self.remaining_s()
        if stage_timeout_s is not None:
            timeout = min(timeout, stage_timeout_s)
        try:
            async with asyncio.timeout(timeout):
                yield
        except TimeoutError as exc:
            if isinstance(exc, DeadlineExceeded):
                raise
            raise DeadlineExceeded(
                f"stage {stage or '?'} timed out after {timeout:.3f}s", stage
            ) from exc

    def snapshot(self) -> dict[str, float]:
        return {
            "llm_calls": self.llm_calls,
            "tokens": self.tokens,
            "agent_steps": self.agent_steps,
            "tool_calls": self.tool_calls,
            "usd": round(self.usd, 6),
            "elapsed_s": round(self.elapsed_s, 3),
        }


async def with_timeout(awaitable: Awaitable[T], timeout_s: float, *, stage: str = "") -> T:
    """Await with a timeout, raising :class:`DeadlineExceeded` (a ``TimeoutError``)."""
    if timeout_s <= 0:
        raise DeadlineExceeded(f"no time left for stage {stage or '?'}", stage)
    try:
        async with asyncio.timeout(timeout_s):
            return await awaitable
    except TimeoutError as exc:
        if isinstance(exc, DeadlineExceeded):
            raise
        raise DeadlineExceeded(f"stage {stage or '?'} timed out after {timeout_s}s", stage) from exc


async def with_fallback(
    awaitable: Awaitable[T], timeout_s: float, fallback: T, *, stage: str = ""
) -> T:
    """Optional-stage helper: on timeout return ``fallback`` (e.g. skip rerank, keep fused order)."""
    try:
        return await with_timeout(awaitable, timeout_s, stage=stage)
    except DeadlineExceeded:
        logger.warning("optional stage %s timed out; using fallback", stage or "?")
        return fallback
