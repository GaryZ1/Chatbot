"""Process-level wiring: one object that owns components + both planes."""

from __future__ import annotations

from typing import Any

from chatbot.build import Components, build_components
from chatbot.config import Settings
from chatbot.core.types import Principal
from chatbot.evaluation.dataset import GoldenExample
from chatbot.evaluation.runner import EvalTrace, SystemUnderTest
from chatbot.ingestion import IngestionPipeline
from chatbot.llm import LLM
from chatbot.pipeline import RAGPipeline, RetrievalResult


class ChatbotService:
    def __init__(self, components: Components) -> None:
        self.components = components
        self.rag = RAGPipeline(components)
        self.ingestion = IngestionPipeline(components)

    @classmethod
    def from_settings(cls, settings: Settings, *, llm: LLM | None = None) -> ChatbotService:
        return cls(build_components(settings, llm=llm))

    async def aclose(self) -> None:
        await self.components.llm.aclose()

    # ------------------------------------------------------------------ evaluation adapter

    def eval_system(self, principal: Principal) -> SystemUnderTest:
        """Return a ``SystemUnderTest`` callable for :func:`chatbot.evaluation.run_eval`.

        Multi-turn examples replay their ``history`` into a fresh session first.
        """

        async def system(example: GoldenExample) -> EvalTrace:
            session_id: str | None = None
            if example.history:
                session = await self.components.memory.load(None, principal)
                for turn in example.history:
                    await self.components.memory.add_turn(session, turn.role, turn.content)
                session_id = session.session_id
            capture: dict[str, Any] = {}
            answer, _ = await self.rag.chat(
                example.question, principal, session_id=session_id, capture=capture
            )
            res: RetrievalResult | None = capture.get("retrieval")
            trace = EvalTrace.from_answer(
                answer,
                candidates=res.candidates if res else (),
                final=res.reranked if res else (),
                context=[h.chunk for h in res.context] if res else (),
            )
            return trace

        return system
