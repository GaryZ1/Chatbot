"""Deterministic LLM for tests, offline evaluation dry-runs and local development.

``FakeLLM`` answers with a user-supplied handler, or a FIFO script of canned
responses. ``parse`` validates the handler's text (JSON) against the schema, so
structured-output code paths are exercised exactly as in production.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable

from chatbot.llm.base import LLM, LLMError, LLMRequest, LLMResponse, T

Handler = Callable[[LLMRequest], "str | LLMResponse"]


def last_user_text(request: LLMRequest) -> str:
    for m in reversed(request.messages):
        if m.role != "user":
            continue
        if isinstance(m.content, str):
            return m.content
        return "\n".join(
            str(b.get("text") or b.get("content") or "") for b in m.content if isinstance(b, dict)
        )
    return ""


class FakeLLM(LLM):
    def __init__(
        self,
        handler: Handler | None = None,
        script: Iterable[str | LLMResponse] | None = None,
        default: str = "I don't know.",
    ) -> None:
        self.handler = handler
        self.script: deque[str | LLMResponse] = deque(script or [])
        self.default = default
        self.calls: list[LLMRequest] = []

    def _respond(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self.script:
            out = self.script.popleft()
        elif self.handler is not None:
            out = self.handler(request)
        else:
            out = self.default
        if isinstance(out, LLMResponse):
            return out
        return LLMResponse(text=out, model="fake", raw_content=[{"type": "text", "text": out}])

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return self._respond(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        text = self._respond(request).text
        for i in range(0, len(text), 16):
            yield text[i : i + 16]

    async def parse(self, request: LLMRequest, schema: type[T]) -> T:
        text = self._respond(request).text
        try:
            return schema.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"fake structured output invalid for {schema.__name__}: {exc}") from exc


_SOURCE_RE = re.compile(r'<source id="(\d+)"[^>]*>\s*(.*?)</source>', re.S)


def extractive_handler(request: LLMRequest) -> str:
    """Offline stand-in for generation used by CI: answers with the first sentence of
    source [1] and cites it; abstains when there are no sources. Structured-output
    calls receive non-JSON and therefore exercise every component's LLM-failure fallback.
    """
    text = last_user_text(request)
    m = _SOURCE_RE.search(text)
    if not m:
        return "I don't know based on the available sources."
    body = re.sub(r"\s+", " ", m.group(2)).strip()
    sentence = re.split(r"(?<=[.!?])\s", body, maxsplit=1)[0][:400]
    return f"{sentence} [{m.group(1)}]"
