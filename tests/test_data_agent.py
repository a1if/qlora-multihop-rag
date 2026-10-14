"""Tool-use traces for the agent: structure, targets, and parity with the live agent's prompts."""

import json
import random

import pytest

pytest.importorskip("faiss")
pytest.importorskip("mcp")
pytest.importorskip("langgraph")

import anyio  # noqa: E402

from src.agent import LocalToolChatModel, build_agent, load_mcp_tools, parse_tool_calls, run_agent  # noqa: E402
from src.data_agent import build_examples, build_trace  # noqa: E402
from src.mcp_server import create_server  # noqa: E402
from src.text_utils import ABSTAIN, extract_answer  # noqa: E402
from tests.test_data_sft import ROWS  # noqa: E402


class JsonTokenizer:
    def apply_chat_template(self, messages, tools=None, **kwargs):
        return json.dumps({"messages": messages, "tools": [t["function"] for t in tools or []]})


class TokenizerOnly:
    tokenizer = JsonTokenizer()


def kinds(trace):
    out = []
    for _, reply in trace:
        msg = parse_tool_calls(reply)
        out.append(("search", msg.tool_calls[0]["args"]["query"]) if msg.tool_calls else ("answer", extract_answer(reply)))
    return out


def test_full_trace_searches_question_then_bridge_then_answers():
    trace = build_trace(ROWS[0], random.Random(0), early_stop_ratio=0, abstain_ratio=0)
    assert kinds(trace) == [("search", ROWS[0]["question"]), ("search", "Seth MacFarlane"), ("answer", "Seth MacFarlane")]
    # the bridge search's results contain the hop-2 paragraph, as seen in the next step's history
    final_history = trace[2][0]
    assert "Seth MacFarlane: Seth MacFarlane sentence one." in final_history[-1].content


def test_early_stop_and_abstain_variants():
    early = build_trace(ROWS[0], random.Random(0), early_stop_ratio=1, abstain_ratio=0)
    assert [k for k, _ in kinds(early)] == ["search", "answer"]
    abstain = build_trace(ROWS[0], random.Random(0), early_stop_ratio=0, abstain_ratio=1)
    assert kinds(abstain)[-1] == ("answer", ABSTAIN)
    assert "Seth MacFarlane: Seth" not in abstain[-1][0][-1].content


def test_examples_are_prerendered_agent_steps():
    model = LocalToolChatModel(llm=TokenizerOnly())
    examples = build_examples(ROWS[:20], model, seed=0)
    assert all(e.task == "agent" and e.prerendered for e in examples)
    assert 40 <= len(examples) <= 60                       # 2-3 steps per question
    first = json.loads(examples[0].instruction)["messages"]
    assert [m["role"] for m in first] == ["system", "user"]


def test_first_step_prompt_matches_live_agent(pipeline):
    """The training prompt for step 1 must equal what the real LangGraph agent sends."""
    server = create_server(pipeline.retriever, pipeline.config, lambda: pipeline)
    sent = []

    class Recorder:
        tokenizer = JsonTokenizer()

        def complete(self, prompt, max_new_tokens, *args):
            sent.append(prompt)
            return "Answer: x"

    async def go():
        from mcp.client import Client

        async with Client(server) as client:
            tools = await load_mcp_tools(client)
            await run_agent(build_agent(LocalToolChatModel(llm=Recorder()), tools), ROWS[0]["question"])
            return tools

    tools = anyio.run(go)
    model = LocalToolChatModel(llm=TokenizerOnly()).bind_tools(tools)
    first = next(e for e in build_examples(ROWS[:1], model, seed=0) if e.id.endswith("#0"))
    assert first.instruction == sent[0]
    assert "k" not in json.loads(first.instruction)["tools"][0]["parameters"]["properties"]   # k hidden from the model
