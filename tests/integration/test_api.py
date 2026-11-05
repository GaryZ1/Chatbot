import json

import pytest
from fastapi.testclient import TestClient

from chatbot.api.app import create_app
from chatbot.api.auth import Authenticator

HEADERS = {"X-User-Id": "alice", "X-User-Groups": "employees"}


@pytest.fixture
def client(service):
    app = create_app(service.components.settings, service=service, authenticator=Authenticator())
    with TestClient(app) as c:
        yield c


def test_health_and_metrics(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["status"] == "ready"
    assert client.get("/metrics").status_code == 200


def test_chat_requires_identity(client):
    assert client.post("/v1/chat", json={"message": "hi"}).status_code == 401


def test_chat_json_and_feedback(client):
    r = client.post(
        "/v1/chat",
        json={"message": "How long does a credit card refund take?"},
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "answered" and body["citations"]
    fb = client.post(
        "/v1/feedback",
        json={"trace_id": body["trace_id"], "rating": "down", "reasons": ["incomplete"]},
        headers=HEADERS,
    )
    assert fb.status_code == 202


def test_chat_streaming_sse(client):
    with client.stream(
        "POST",
        "/v1/chat",
        json={"message": "What is the daily meal allowance?", "stream": True},
        headers=HEADERS,
    ) as r:
        raw = "".join(r.iter_text())
    events = [blk for blk in raw.split("\n\n") if blk.strip()]
    assert events[-1].startswith("event: answer")
    final = json.loads(events[-1].split("data: ", 1)[1])
    assert final["session_id"]


def test_session_isolation(client):
    r = client.post("/v1/chat", json={"message": "hi"}, headers=HEADERS)
    sid = r.json()["session_id"]
    other = {"X-User-Id": "mallory", "X-User-Groups": "employees"}
    r2 = client.post("/v1/chat", json={"message": "hi", "session_id": sid}, headers=other)
    assert r2.status_code == 403


def test_ingest_requires_admin(client):
    r = client.post("/v1/ingest", json={"paths": ["data"]}, headers=HEADERS)
    assert r.status_code == 403
