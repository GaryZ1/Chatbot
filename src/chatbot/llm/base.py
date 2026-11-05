"""Provider-neutral LLM interface.

Models are *tiered by route* (docs/10-generation-context.md §4, docs/14-production-llmops.md §2):
``small`` for condensing / routing / grading, ``mid`` for most answers, ``frontier``
for agentic planning and hard synthesis. Concrete clients map tiers to model ids.

Message ``content`` uses the Anthropic wire format (``str`` or a list of content
block dicts: ``text`` / ``tool_use`` / ``tool_result``), which is expressive enough
for tool-using agents and easy to translate for other providers.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class Tier(StrEnum):
    SMALL = "small"
    MID = "mid"
    FRONTIER = "frontier"


class LLMMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
        )


class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    #: Stable, cacheable prefix. Never put timestamps or per-request ids here.
    system: str = ""
    tier: Tier = Tier.MID
    #: Thinking tokens count against this on current models, so do not lowball it.
    max_tokens: int = 16000
    tools: list[ToolSpec] = Field(default_factory=list)
    #: Effort for models that support it (low | medium | high | xhigh | max).
    effort: str | None = None
    #: Mark the system prompt as a cache breakpoint (prompt caching).
    cache_system: bool = True
    stop_sequences: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def simple(cls, prompt: str, *, system: str = "", tier: Tier = Tier.MID, **kw: Any) -> LLMRequest:
        return cls(messages=[LLMMessage(role="user", content=prompt)], system=system, tier=tier, **kw)


class LLMResponse(BaseModel):
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str = "end_turn"
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    #: Assistant content blocks exactly as returned; append these back for tool loops.
    raw_content: list[dict[str, Any]] = Field(default_factory=list)


class LLMError(RuntimeError):
    """Raised for non-retryable provider failures or exhausted retries."""


class LLMRefusal(LLMError):
    """The model (and any fallback chain) declined the request."""


class LLM(abc.ABC):
    """Async LLM client. Implementations must be safe for concurrent use."""

    @abc.abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    @abc.abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Yield text deltas. Implemented as an async generator."""

    @abc.abstractmethod
    async def parse(self, request: LLMRequest, schema: type[T]) -> T:
        """Structured output validated against a Pydantic model."""

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None
