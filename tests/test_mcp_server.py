"""MCP server driven by a real MCP client over the in-process transport."""

import json

import pytest

pytest.importorskip("faiss")
pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp.client import Client  # noqa: E402

from src.mcp_server import create_server  # noqa: E402
from tests.conftest import QUESTION  # noqa: E402


def call(server, tool, args):
    async def go():
        async with Client(server, raise_exceptions=True) as client:
            return await client.call_tool(tool, args)
    return anyio.run(go)


def payload(result):
    """Structured tool output if present, else the JSON text content."""
    if result.structured_content is not None:
        data = result.structured_content
        return data.get("result", data)
    return json.loads(result.content[0].text)


def test_lists_tools(pipeline):
    async def go():
        async with Client(create_server(pipeline)) as client:
            return [t.name for t in (await client.list_tools()).tools]
    assert set(anyio.run(go)) == {"search", "get_passage", "answer"}


def test_search_tool(pipeline):
    hits = payload(call(create_server(pipeline), "search", {"query": "Where was Marie Curie born?", "k": 2}))
    assert len(hits) == 2 and hits[0]["title"] == "Marie Curie"
    assert {"id", "title", "text", "score"} <= set(hits[0])


def test_get_passage_tool(pipeline):
    doc = payload(call(create_server(pipeline), "get_passage", {"passage_id": 1}))
    assert doc["title"] == "Marie Curie" and doc["formatted"].startswith("Marie Curie: ")


def test_answer_tool_runs_multihop_pipeline(pipeline):
    out = payload(call(create_server(pipeline), "answer", {"question": QUESTION, "mode": "iterative"}))
    assert out["answer"] == "Poland"
    assert out["queries"] == [QUESTION, "Marie Curie birthplace"]
    assert any(p["title"] == "Marie Curie" for p in out["passages"])


def test_invalid_arguments_are_rejected(pipeline):
    result = call(create_server(pipeline), "search", {"query": "x", "k": 99})
    assert result.is_error


def test_unknown_passage_is_an_error(pipeline):
    assert call(create_server(pipeline), "get_passage", {"passage_id": 12345}).is_error
