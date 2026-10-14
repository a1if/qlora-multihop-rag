import pytest

pytest.importorskip("faiss")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src.api import create_app  # noqa: E402
from tests.conftest import QUESTION  # noqa: E402


@pytest.fixture
def client(pipeline, config):
    with TestClient(create_app(pipeline=pipeline, config=config)) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["llm"] == "fake"


def test_answer_endpoint(client):
    resp = client.post("/v1/answer", json={"question": QUESTION, "mode": "iterative", "answer_style": "short"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Poland"
    assert body["mode"] == "iterative"
    assert any(p["title"] == "Marie Curie" for p in body["passages"])
    assert body["timings"]["llm_calls"] == 2
    assert resp.headers["x-request-id"] == body["request_id"]


def test_request_id_is_propagated(client):
    resp = client.post("/v1/answer", json={"question": QUESTION}, headers={"x-request-id": "abc123"})
    assert resp.json()["request_id"] == "abc123"


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": QUESTION, "mode": "bogus"},
    {"question": QUESTION, "top_k": 0},
    {"question": QUESTION, "hops": 9},
])
def test_validation_errors(client, payload):
    assert client.post("/v1/answer", json=payload).status_code == 422


def test_metrics_exposed(client):
    client.post("/v1/answer", json={"question": QUESTION, "mode": "single"})
    text = client.get("/metrics").text
    assert 'qmr_requests_total{mode="single",status="ok"} 1.0' in text
    assert "qmr_stage_seconds_bucket" in text


def test_backend_failure_returns_502(pipeline, config):
    class BrokenLLM:
        name = "broken"

        def generate(self, *args, **kwargs):
            raise ConnectionError("upstream down")

        def close(self):
            pass

    pipeline.llm = BrokenLLM()
    with TestClient(create_app(pipeline=pipeline, config=config)) as c:
        resp = c.post("/v1/answer", json={"question": QUESTION})
        assert resp.status_code == 502
        assert 'status="error"' in c.get("/metrics").text
