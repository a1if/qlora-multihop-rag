"""LangGraph agent over the MCP tools, with a scripted model instead of Qwen3."""

import json

import pytest

pytest.importorskip("faiss")
pytest.importorskip("mcp")
pytest.importorskip("langgraph")

import anyio  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from mcp.client import Client  # noqa: E402

from src.agent import (  # noqa: E402
    LocalToolChatModel, build_agent, load_mcp_tools, parse_tool_calls, run_agent, to_chat_dicts,
)
from src.mcp_server import create_server  # noqa: E402
from tests.conftest import QUESTION  # noqa: E402


def test_parse_qwen_tool_call():
    msg = parse_tool_calls('<tool_call>\n{"name": "search", "arguments": {"query": "Marie Curie", "k": 3}}\n</tool_call>')
    assert msg.tool_calls[0]["name"] == "search"
    assert msg.tool_calls[0]["args"] == {"query": "Marie Curie", "k": 3}
    assert msg.content == ""


def test_parse_final_answer_and_bare_json_fallback():
    assert parse_tool_calls("<think>done</think>Answer: Poland").tool_calls == []
    bare = parse_tool_calls('{"name": "search", "arguments": {"query": "x"}}')   # tags stripped as special tokens
    assert bare.tool_calls[0]["args"] == {"query": "x"}
    assert parse_tool_calls('<tool_call>{"name": "search", "arguments": oops}</tool_call>').tool_calls == []


def test_to_chat_dicts_round_trips_tool_calls():
    msgs = [SystemMessage("sys"), HumanMessage("q"),
            AIMessage("", tool_calls=[{"name": "search", "args": {"query": "q"}, "id": "c1"}]),
            ToolMessage("[1] A: text", tool_call_id="c1")]
    dicts = to_chat_dicts(msgs)
    assert [d["role"] for d in dicts] == ["system", "user", "assistant", "tool"]
    assert dicts[2]["tool_calls"][0]["function"] == {"name": "search", "arguments": {"query": "q"}}


class ScriptedTokenizer:
    def apply_chat_template(self, messages, tools=None, **kwargs):
        return json.dumps({"messages": messages, "tools": [t["function"]["name"] for t in tools or []]})


class ScriptedLLM:
    """Searches the question, then the bridge entity, then answers from what it saw."""

    name = "scripted"

    def __init__(self):
        self.tokenizer = ScriptedTokenizer()
        self.prompts = []

    def complete(self, prompt, max_new_tokens, *args):
        self.prompts.append(json.loads(prompt))
        tool_turns = [m for m in self.prompts[-1]["messages"] if m["role"] == "tool"]
        if len(tool_turns) == 0:
            return f'<tool_call>{{"name": "search", "arguments": {{"query": "{QUESTION}", "k": 2}}}}</tool_call>'
        if len(tool_turns) == 1:
            return '<tool_call>{"name": "search", "arguments": {"query": "Marie Curie", "k": 2}}</tool_call>'
        seen = " ".join(m["content"] for m in tool_turns)
        return "Answer: Poland" if "born in Warsaw" in seen else "Answer: insufficient evidence"


def test_agent_runs_two_hops_over_mcp(pipeline):
    server = create_server(pipeline.retriever, pipeline.config, lambda: pipeline)
    llm = ScriptedLLM()

    async def go():
        async with Client(server) as client:
            tools = await load_mcp_tools(client)
            assert [t.name for t in tools] == ["search", "get_passage"]      # `answer` is not given to the agent
            return await run_agent(build_agent(LocalToolChatModel(llm=llm), tools), QUESTION)

    result = anyio.run(go)
    assert result["answer"] == "Poland"
    assert result["queries"] == [QUESTION, "Marie Curie"]
    assert "Marie Curie" in result["retrieved_titles"]
    assert result["llm_calls"] == 3 and not result["stopped_at_limit"]
    assert llm.prompts[0]["tools"] == ["search", "get_passage"]              # tools rendered into the prompt


def test_tool_errors_are_returned_to_the_model(pipeline):
    server = create_server(pipeline.retriever, pipeline.config, lambda: pipeline)

    async def go():
        async with Client(server) as client:
            tools = {t.name: t for t in await load_mcp_tools(client)}
            return await tools["get_passage"].ainvoke(
                {"type": "tool_call", "name": "get_passage", "args": {"passage_id": 999}, "id": "c1"})

    msg = anyio.run(go)
    assert msg.content.startswith("ERROR:") and "Valid ids are 0-4" in msg.content


def test_step_limit_stops_runaway_agents(pipeline):
    class LoopingLLM(ScriptedLLM):
        def complete(self, prompt, max_new_tokens, *args):
            return '<tool_call>{"name": "search", "arguments": {"query": "polonium"}}</tool_call>'

    server = create_server(pipeline.retriever, pipeline.config, lambda: pipeline)

    async def go():
        async with Client(server) as client:
            agent = build_agent(LocalToolChatModel(llm=LoopingLLM()), await load_mcp_tools(client))
            return await run_agent(agent, QUESTION, max_steps=3)

    result = anyio.run(go)
    assert result["stopped_at_limit"] and result["answer"] == ""
