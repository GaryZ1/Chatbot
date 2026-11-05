"""Model gateway: ordered provider failover (docs/14-production-llmops.md §5).

Refusals are *not* retried on another provider: a policy decline is a decision,
not an outage, and server-side refusal fallbacks already ran inside the call.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence

from chatbot.llm.base import LLM, LLMError, LLMRefusal, LLMRequest, LLMResponse, T

log = logging.getLogger(__name__)


class FailoverLLM(LLM):
    def __init__(self, providers: Sequence[LLM]) -> None:
        if not providers:
            raise ValueError("FailoverLLM needs at least one provider")
        self.providers = list(providers)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        last: Exception | None = None
        for p in self.providers:
            try:
                return await p.generate(request)
            except LLMRefusal:
                raise
            except LLMError as exc:
                log.warning("provider %s failed, failing over: %s", type(p).__name__, exc)
                last = exc
        raise LLMError(f"all providers failed: {last}")

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        last: Exception | None = None
        for p in self.providers:
            started = False
            try:
                async for delta in p.stream(request):
                    started = True
                    yield delta
                return
            except LLMRefusal:
                raise
            except LLMError as exc:
                if started:  # cannot transparently fail over a partially streamed answer
                    raise
                log.warning("provider %s stream failed, failing over: %s", type(p).__name__, exc)
                last = exc
        raise LLMError(f"all providers failed: {last}")

    async def parse(self, request: LLMRequest, schema: type[T]) -> T:
        last: Exception | None = None
        for p in self.providers:
            try:
                return await p.parse(request, schema)
            except LLMRefusal:
                raise
            except LLMError as exc:
                last = exc
        raise LLMError(f"all providers failed: {last}")

    async def aclose(self) -> None:
        for p in self.providers:
            await p.aclose()
