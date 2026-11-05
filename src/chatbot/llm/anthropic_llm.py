"""Claude implementation of :class:`~chatbot.llm.base.LLM` (official ``anthropic`` SDK).

* Tiered models: ``small`` -> Claude Haiku 5.5, ``mid`` / ``frontier`` -> Claude Opus 5.5
  at different effort levels (configurable).
* Prompt caching: the system prompt is a cache breakpoint, so the stable prefix
  (role, answer policy, safety rules) is served from cache (docs/10 §2 rule 1).
* Server-side refusal fallbacks are enabled for models that support them.
* Structured outputs use ``messages.parse`` against a Pydantic model.
* Thinking is left adaptive (the default on current models); depth is set by effort.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from chatbot.llm.base import (
    LLM,
    LLMError,
    LLMRefusal,
    LLMRequest,
    LLMResponse,
    T,
    Tier,
    ToolCall,
    Usage,
)

log = logging.getLogger(__name__)

DEFAULT_MODELS: dict[Tier, str] = {
    Tier.SMALL: "claude-haiku-5-5",
    Tier.MID: "claude-opus-5-5",
    Tier.FRONTIER: "claude-opus-5-5",
}
DEFAULT_EFFORT: dict[Tier, str] = {Tier.SMALL: "low", Tier.MID: "medium", Tier.FRONTIER: "high"}

#: Models that accept the server-side ``fallbacks="default"`` refusal routing.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM(LLM):
    def __init__(
        self,
        *,
        models: dict[Tier, str] | None = None,
        effort: dict[Tier, str] | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        enable_fallbacks: bool = True,
        client: Any | None = None,
    ) -> None:
        if client is None:
            import anthropic  # optional dependency: pip install chatbot[anthropic]

            kwargs: dict[str, Any] = {"timeout": timeout_s, "max_retries": max_retries}
            if api_key:
                kwargs["api_key"] = api_key
            client = anthropic.AsyncAnthropic(**kwargs)
        self._client = client
        self.models = {**DEFAULT_MODELS, **(models or {})}
        self.effort = {**DEFAULT_EFFORT, **(effort or {})}
        self.enable_fallbacks = enable_fallbacks

    # ------------------------------------------------------------------ request building

    def _params(self, req: LLMRequest) -> dict[str, Any]:
        model = self.models[req.tier]
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": req.max_tokens,
            "messages": [m.model_dump() for m in req.messages],
        }
        if req.system:
            block: dict[str, Any] = {"type": "text", "text": req.system}
            if req.cache_system:
                block["cache_control"] = {"type": "ephemeral"}
            params["system"] = [block]
        if req.tools:
            params["tools"] = [t.model_dump() for t in req.tools]
        effort = req.effort or self.effort.get(req.tier)
        if effort:
            params["output_config"] = {"effort": effort}
        if req.stop_sequences:
            params["stop_sequences"] = req.stop_sequences
        betas: list[str] = []
        if self.enable_fallbacks and model in _FALLBACK_MODELS:
            betas.append(_FALLBACK_BETA)
            params["fallbacks"] = "default"
        if betas:
            params["betas"] = betas
        return params

    @staticmethod
    def _to_response(msg: Any) -> LLMResponse:
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusal(f"model declined the request (category={category})")
        texts: list[str] = []
        calls: list[ToolCall] = []
        raw: list[dict[str, Any]] = []
        for block in msg.content:
            raw.append(block.model_dump(exclude_none=True))
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        u = msg.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        return LLMResponse(
            text="".join(texts),
            tool_calls=calls,
            stop_reason=msg.stop_reason or "end_turn",
            model=msg.model,
            usage=usage,
            raw_content=raw,
        )

    @staticmethod
    def _wrap(exc: Exception) -> LLMError:
        import anthropic

        if isinstance(exc, anthropic.RateLimitError):
            return LLMError(f"rate limited after retries: {exc}")
        if isinstance(exc, anthropic.APIStatusError):
            return LLMError(f"API error {exc.status_code}: {exc.message}")
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMError(f"connection error: {exc}")
        return LLMError(str(exc))

    # ------------------------------------------------------------------ LLM API

    async def generate(self, request: LLMRequest) -> LLMResponse:
        import anthropic

        try:
            msg = await self._client.beta.messages.create(**self._params(request))
        except anthropic.APIError as exc:
            raise self._wrap(exc) from exc
        return self._to_response(msg)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        import anthropic

        try:
            async with self._client.beta.messages.stream(**self._params(request)) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except anthropic.APIError as exc:
            raise self._wrap(exc) from exc
        if final.stop_reason == "refusal":
            raise LLMRefusal("model declined the request mid-stream")

    async def parse(self, request: LLMRequest, schema: type[T]) -> T:
        import anthropic

        params = self._params(request)
        try:
            msg = await self._client.beta.messages.parse(output_format=schema, **params)
        except anthropic.APIError as exc:
            raise self._wrap(exc) from exc
        if msg.stop_reason == "refusal":
            raise LLMRefusal("model declined the structured-output request")
        parsed = getattr(msg, "parsed_output", None)
        if parsed is None:
            raise LLMError(f"structured output missing (stop_reason={msg.stop_reason})")
        return parsed

    async def aclose(self) -> None:
        await self._client.close()
