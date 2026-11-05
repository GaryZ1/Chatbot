"""FastAPI service: ``/v1/chat`` (JSON or SSE streaming), feedback, ingestion,
memory controls, health and Prometheus metrics (docs/14-production-llmops.md §5).

Run: ``uvicorn chatbot.api.app:create_app --factory``.
"""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import PlainTextResponse, StreamingResponse

from chatbot.api.auth import Authenticator
from chatbot.api.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    IngestRequest,
    IngestResponse,
)
from chatbot.config import Settings
from chatbot.core.types import Answer, Principal
from chatbot.ops.feedback import Feedback
from chatbot.service import ChatbotService

log = logging.getLogger(__name__)

ADMIN_ROLE = "admin"


def _to_response(answer: Answer, session_id: str) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        answer=answer.text,
        status=answer.status,
        citations=answer.citations,
        route=answer.route,
        trace_id=answer.trace_id,
        latency_ms=answer.latency_ms,
    )


def create_app(
    settings: Settings | None = None,
    *,
    service: ChatbotService | None = None,
    authenticator: Authenticator | None = None,
) -> FastAPI:
    settings = settings or Settings.load()
    auth = authenticator or Authenticator.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or ChatbotService.from_settings(settings)
        try:
            yield
        finally:
            await app.state.service.aclose()

    app = FastAPI(title="Chatbot", version="0.1.0", lifespan=lifespan)

    def get_service(request: Request) -> ChatbotService:
        svc: ChatbotService = request.app.state.service
        return svc

    Svc = Annotated[ChatbotService, Depends(get_service)]
    User = Annotated[Principal, Depends(auth)]

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(svc: Svc) -> dict[str, Any]:
        n = await svc.components.vector_store.count()
        return {"status": "ready" if n else "empty_index", "chunks": n}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(svc: Svc) -> str:
        return svc.components.metrics.render_prometheus()

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, user: User, svc: Svc) -> Any:
        try:
            if not req.stream:
                answer, session = await svc.rag.chat(
                    req.message, user, session_id=req.session_id, filters=req.filters or None
                )
                return _to_response(answer, session.session_id)
            # Load the session eagerly so ownership errors become a 403, not a broken stream.
            await svc.components.memory.load(req.session_id, user)
        except PermissionError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "session not accessible") from exc

        async def events() -> AsyncIterator[bytes]:
            try:
                async for ev in svc.rag.chat_stream(
                    req.message,
                    user,
                    session_id=req.session_id,
                    filters=req.filters or None,
                    stream=True,
                ):
                    if ev.type == "delta":
                        yield f"event: delta\ndata: {json.dumps({'text': ev.text})}\n\n".encode()
                    elif ev.answer is not None and ev.session is not None:
                        payload = _to_response(ev.answer, ev.session.session_id).model_dump_json()
                        yield f"event: answer\ndata: {payload}\n\n".encode()
            except Exception:
                log.exception("stream failed")
                yield b'event: error\ndata: {"error": "internal error"}\n\n'

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/feedback", status_code=status.HTTP_202_ACCEPTED)
    async def feedback(req: FeedbackRequest, user: User, svc: Svc) -> dict[str, str]:
        record = await svc.components.trace_store.get(req.trace_id)
        if record is not None and record.user_id and record.user_id != user.user_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown trace")
        fb = Feedback(
            trace_id=req.trace_id,
            user_id=user.user_id,
            rating=req.rating,
            reasons=req.reasons,
            comment=req.comment,
            correction=req.correction,
        )
        await svc.components.feedback.add(fb)
        await svc.components.trace_store.attach_feedback(req.trace_id, fb.model_dump(mode="json"))
        return {"status": "accepted"}

    @app.post("/v1/ingest", response_model=IngestResponse)
    async def ingest(req: IngestRequest, user: User, svc: Svc) -> Any:
        if ADMIN_ROLE not in user.roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "admin role required")
        allowed_root = Path(os.environ.get("CHATBOT_INGEST_ROOT", "data")).resolve()
        for raw in req.paths:
            resolved = Path(raw).resolve()
            if resolved != allowed_root and allowed_root not in resolved.parents:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, f"path must be inside {allowed_root}"
                )
        try:
            report = await svc.ingestion.ingest_paths(req.paths, full=not req.incremental)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        return IngestResponse(
            added=report.added,
            updated=report.updated,
            deleted=report.deleted,
            skipped=report.skipped,
            chunks=report.chunks,
            quarantined=report.quarantined,
            failed=report.failed,
        )

    @app.get("/v1/memories")
    async def list_memories(user: User, svc: Svc) -> list[dict[str, Any]]:
        lt = svc.components.long_term
        if lt is None:
            return []
        return [m.model_dump(mode="json") for m in await lt.list_memories(user)]

    @app.delete("/v1/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_memory(memory_id: str, user: User, svc: Svc) -> None:
        lt = svc.components.long_term
        if lt is None or not await lt.delete(user, memory_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown memory")

    @app.delete("/v1/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_session(session_id: str, user: User, svc: Svc) -> None:
        try:
            await svc.components.memory.load(session_id, user)
        except PermissionError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown session") from exc
        await svc.components.memory.delete(session_id)

    return app
