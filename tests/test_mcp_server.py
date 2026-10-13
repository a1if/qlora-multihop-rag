"""MCP server driven by a real MCP client over the in-process transport."""

import json
import logging

import pytest

pytest.importorskip("faiss")
pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp.client import Client  # noqa: E402

from src.mcp_server import MAX_K, create_server  # noqa: E402
from tests.conftest import QUESTION  # noqa: E402


@pytest.fixture
def server(pipeline):
    return create_server(pipeline.retriever, pipeline.config, lambda: pipeline, llm_name="fake")


def run(server, fn):
    async def go():
        async with Client(server) as client:
            return await fn(client)
    return anyio.run(go)


def call(server, tool, args):
    return run(server, lambda c: c.call_tool(tool, args))


def data(result):
    assert not result.is_error, result.content
    return result.structured_content


def error_text(result):
    assert result.is_error
    return result.content[0].text


def test_lists_three_tools_with_typed_schemas(server):
    tools = {t.name: t for t in run(server, lambda c: c.list_tools()).tools}
    assert set(tools) == {"search", "get_passage", "answer"}
    k = tools["search"].input_schema["properties"]["k"]
    assert (k["minimum"], k["maximum"], k["default"]) == (1, MAX_K, 5)
    assert "description" in k
    assert tools["answer"].input_schema["properties"]["mode"]["enum"] == ["closed", "single", "iterative", "decompose"]
    # Pydantic return types become output schemas
    assert "results" in tools["search"].output_schema["properties"]
    assert "answer" in tools["answer"].output_schema["properties"]


def test_search(server):
    out = data(call(server, "search", {"query": "Where was Marie Curie born?", "k": 2}))
    assert out["query"] == "Where was Marie Curie born?"
    assert len(out["results"]) == 2 and out["results"][0]["title"] == "Marie Curie"
    assert out["results"][0]["score"] is not None


def test_get_passage(server):
    out = data(call(server, "get_passage", {"passage_id": 1}))
    assert out["title"] == "Marie Curie" and out["formatted"].startswith("Marie Curie: ")


def test_answer_runs_multihop_pipeline(server):
    out = data(call(server, "answer", {"question": QUESTION, "mode": "iterative"}))
    assert out["answer"] == "Poland"
    assert out["queries"] == [QUESTION, "Marie Curie birthplace"]
    assert any(p["title"] == "Marie Curie" for p in out["passages"])


@pytest.mark.parametrize("args", [
    {"query": "x", "k": 0}, {"query": "x", "k": MAX_K + 1}, {"query": ""}, {"query": "x" * 501},
])
def test_search_validation(server, args):
    assert call(server, "search", args).is_error


def test_blank_query_has_clear_message(server):
    assert "query is empty" in error_text(call(server, "search", {"query": "   "}))


def test_unknown_passage_message_names_valid_range(server):
    msg = error_text(call(server, "get_passage", {"passage_id": 12345}))
    assert "No passage with id 12345" in msg and "0-4" in msg


def test_answer_failure_is_reported_not_crashed(pipeline):
    def broken():
        raise RuntimeError("CUDA out of memory")

    server = create_server(pipeline.retriever, pipeline.config, broken)
    msg = error_text(call(server, "answer", {"question": QUESTION}))
    assert "CUDA out of memory" in msg and "`search` still works" in msg
    assert not call(server, "search", {"query": "polonium"}).is_error


def test_llm_loads_lazily(pipeline):
    loads = []

    def get():
        loads.append(1)
        return pipeline

    server = create_server(pipeline.retriever, pipeline.config, get)
    call(server, "search", {"query": "polonium"})
    assert loads == []                                   # search never touches the LLM
    call(server, "answer", {"question": QUESTION})
    assert loads == [1]


def test_corpus_resource(server):
    res = run(server, lambda c: c.read_resource("corpus://info"))
    info = json.loads(res.contents[0].text)
    assert info["passages"] == 5 and info["llm"] == "fake"


def test_every_request_is_logged(server, caplog):
    with caplog.at_level(logging.INFO, logger="qmr.mcp"):
        call(server, "search", {"query": "polonium"})
        call(server, "get_passage", {"passage_id": 999})
    events = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
    calls = [e for e in events if e["method"] == "tools/call"]
    assert [(e["tool"], e["outcome"]) for e in calls] == [("search", "ok"), ("get_passage", "tool_error")]
    assert all(e["duration_ms"] >= 0 for e in calls)


def test_background_retriever_waits_then_serves(pipeline):
    import threading

    from src.mcp_server import BackgroundRetriever

    release = threading.Event()

    def slow_load():
        release.wait(5)
        return pipeline.retriever

    bg = BackgroundRetriever(slow_load, timeout_s=5)
    server = create_server(bg, pipeline.config, lambda: pipeline)   # must not block on the index
    release.set()
    out = call(server, "search", {"query": "polonium", "k": 1})
    assert not out.is_error and out.structured_content["results"][0]["title"] == "Polonium"


def test_background_retriever_reports_load_failure(pipeline):
    from src.mcp_server import BackgroundRetriever

    def broken():
        raise FileNotFoundError("index.faiss missing")

    server = create_server(BackgroundRetriever(broken, timeout_s=5), pipeline.config, lambda: pipeline)
    assert "index.faiss missing" in error_text(call(server, "search", {"query": "polonium"}))
