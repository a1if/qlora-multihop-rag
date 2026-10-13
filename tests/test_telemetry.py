"""One MCP tool call should produce a single connected trace down to retrieval and LLM spans."""

import json

import pytest

pytest.importorskip("faiss")
pytest.importorskip("mcp")
pytest.importorskip("opentelemetry.sdk")

import anyio  # noqa: E402
from mcp.client import Client  # noqa: E402

from src import telemetry  # noqa: E402
from src.mcp_server import create_server  # noqa: E402
from tests.conftest import QUESTION  # noqa: E402


@pytest.fixture(scope="module")
def trace_file(tmp_path_factory):
    # The global tracer provider can only be set once per process.
    path = tmp_path_factory.mktemp("traces") / "spans.jsonl"
    mp = pytest.MonkeyPatch()
    mp.setenv("QMR_TRACE_FILE", str(path))
    assert telemetry.setup_tracing("test-service")
    yield path
    mp.undo()


def spans_for(path, tool):
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    root = next(r for r in reversed(records) if r["name"] == f"tools/call {tool}")
    return root, [r for r in records if r["trace_id"] == root["trace_id"]]


def test_answer_call_is_one_connected_trace(trace_file, pipeline):
    server = create_server(pipeline.retriever, pipeline.config, lambda: pipeline)

    async def go():
        async with Client(server) as client:
            await client.call_tool("answer", {"question": QUESTION, "mode": "iterative"})
    anyio.run(go)

    root, spans = spans_for(trace_file, "answer")
    names = [s["name"] for s in spans]
    assert root["attributes"]["gen_ai.tool.name"] == "answer"
    assert names.count("rag.answer") == 1
    assert names.count("retrieval.search") == 2          # one per hop
    assert names.count("llm.generate") == 2              # follow-up query + reader

    by_id = {s["span_id"]: s for s in spans}
    rag = next(s for s in spans if s["name"] == "rag.answer")
    assert by_id[rag["parent_id"]]["name"] == "tools/call answer"
    for s in spans:
        if s["name"] in ("retrieval.search", "llm.generate"):
            assert s["parent_id"] == rag["span_id"]
    assert rag["attributes"]["rag.mode"] == "iterative" and rag["attributes"]["rag.llm_calls"] == 2


def test_summary_table(trace_file):
    rows = {r["span"]: r for r in telemetry.summarize(str(trace_file))}
    assert rows["rag.answer"]["count"] >= 1 and rows["rag.answer"]["p95_ms"] >= 0
