"""
src/agent.py
LangGraph ReAct agent that answers multi-hop questions with the MCP server's
tools, driven by the local (fine-tuned) Qwen3 model. No hosted LLM.

Unlike the fixed pipeline (always search → follow-up search → answer), the
agent decides how many searches to run, what to search for, when to read a
full passage, and when it has enough evidence.

Pieces:
    LocalToolChatModel   LangChain chat model over src.llm.HFLLM: renders the
                         conversation with Qwen3's native tool-calling template and
                         parses <tool_call> blocks back into tool calls; generation
                         goes through the same dynamic batching as the pipeline
    load_mcp_tools       MCP 2.x tools -> LangChain StructuredTools (the official
                         langchain-mcp-adapters still require mcp<2)
    build_agent          langgraph.prebuilt.create_react_agent over the two

Usage:
    python -m src.agent "Glenn Quagmire is voiced by the creator of what 2012 film?" \\
        --llm hf:D:/qmr/runs/sft_full/merged --index results/full/index_n500_seed0
    python -m src.agent --eval --n 500 --llm hf:Qwen/Qwen3-1.7B --name base_agent ...
"""

import argparse
import json
import os
import re
import time
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from src.telemetry import span
from src.text_utils import extract_answer, strip_reasoning

SYSTEM_PROMPT = (
    "You answer questions by searching a Wikipedia corpus with the `search` tool.\n"
    "Questions often need two facts. Search for the question first. If a passage names a person, "
    "work or place you still need facts about, search for that name next. Use `get_passage` only if "
    "a preview is cut off before the fact you need. Use at most {max_searches} searches.\n"
    "When you know the answer, reply with exactly one line: 'Answer: <short answer>' (an entity, "
    "date, number, or yes/no). If the passages do not contain the answer, reply "
    "'Answer: insufficient evidence'."
)

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_BARE_CALL = re.compile(r"\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{.*?\}\s*\}", re.S)


# ─── Chat Model ───────────────────────────────────────────────────────────────

def to_chat_dicts(messages: list[BaseMessage]) -> list[dict]:
    """LangChain messages -> the dicts HF chat templates expect (OpenAI-style tool calls)."""
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msg = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                msg["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
                                     for c in m.tool_calls]
            out.append(msg)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content})
        else:
            raise TypeError(f"Unsupported message type {type(m).__name__}")
    return out


def parse_tool_calls(text: str) -> AIMessage:
    """Model output -> AIMessage with tool_calls (Qwen3 emits <tool_call>{json}</tool_call>)."""
    text = strip_reasoning(text)
    blocks = _TOOL_CALL.findall(text) or _BARE_CALL.findall(text)
    calls = []
    for block in blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        args = data.get("arguments", {})
        if isinstance(args, str):  # some templates double-encode
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if isinstance(data.get("name"), str) and isinstance(args, dict):
            calls.append({"name": data["name"], "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"})
    content = _TOOL_CALL.sub("", text).strip() if calls else text.strip()
    return AIMessage(content=content, tool_calls=calls)


class LocalToolChatModel(BaseChatModel):
    """Tool-calling chat model over an in-process HFLLM (uses its tokenizer's chat template)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Any
    max_new_tokens: int = 160
    tool_schemas: list[dict] = []

    @property
    def _llm_type(self) -> str:
        return "local-hf-tool-chat"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tool_schemas": [convert_to_openai_tool(t) for t in tools]})

    def render(self, messages: list[BaseMessage]) -> str:
        return self.llm.tokenizer.apply_chat_template(
            to_chat_dicts(messages), tools=self.tool_schemas or None, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = self.render(messages)
        with span("llm.generate", gen_ai__request__model=getattr(self.llm, "name", None),
                  gen_ai__request__max_tokens=self.max_new_tokens, llm__prompt_chars=len(prompt)):
            text = self.llm.complete(prompt, self.max_new_tokens)
        return ChatResult(generations=[ChatGeneration(message=parse_tool_calls(text))])


# ─── MCP Tools ────────────────────────────────────────────────────────────────

def format_tool_result(name: str, data: dict) -> str:
    """Compact text for the model: numbered passages for search, plain text otherwise."""
    if name == "search":
        return "\n".join(f"[{p['id']}] {p.get('title')}: {p['text']}" for p in data.get("results", [])) or "No results."
    if name == "get_passage":
        return data.get("formatted", json.dumps(data))
    return json.dumps(data, ensure_ascii=False)


async def load_mcp_tools(client, allowed: tuple[str, ...] = ("search", "get_passage")) -> list[StructuredTool]:
    """Wrap an open MCP client's tools as LangChain tools (content for the model, artifact = structured result)."""
    from opentelemetry.propagate import inject

    tools = []
    for t in (await client.list_tools()).tools:
        if t.name not in allowed:
            continue

        async def call(_name=t.name, **kwargs):
            meta: dict = {}
            inject(meta)                       # continue the agent's trace inside the server
            result = await client.call_tool(_name, kwargs, meta=meta or None)
            if result.is_error:
                return f"ERROR: {result.content[0].text}", {"error": True}
            data = result.structured_content or {}
            return format_tool_result(_name, data), data

        tools.append(StructuredTool(name=t.name, description=t.description or t.name, args_schema=t.input_schema,
                                    coroutine=call, response_format="content_and_artifact"))
    return tools


def build_agent(chat_model: BaseChatModel, tools: list, max_searches: int = 4):
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(chat_model, tools, prompt=SYSTEM_PROMPT.format(max_searches=max_searches))


# ─── Running ──────────────────────────────────────────────────────────────────

async def run_agent(agent, question: str, max_steps: int = 6) -> dict:
    """One question -> answer, tool calls made, passages seen, latency."""
    from langgraph.errors import GraphRecursionError

    start = time.perf_counter()
    with span("agent.run", agent__question_chars=len(question)) as s:
        try:
            state = await agent.ainvoke({"messages": [HumanMessage(question)]},
                                        config={"recursion_limit": 2 * max_steps + 1})
            messages, stopped = state["messages"], False
        except GraphRecursionError:
            messages, stopped = [], True
        calls = [c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
        seen = []
        for m in messages:
            if isinstance(m, ToolMessage) and isinstance(m.artifact, dict):
                passages = m.artifact.get("results") or ([m.artifact] if "title" in m.artifact else [])
                seen += [p["title"] for p in passages if p.get("title") not in seen]
        final = next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), "")
        s.set_attributes({"agent.tool_calls": len(calls), "agent.stopped_at_limit": stopped})
    return {
        "answer": extract_answer(final) if final else "",
        "raw": final,
        "tool_calls": [{"name": c["name"], "args": c["args"]} for c in calls],
        "queries": [c["args"].get("query") for c in calls if c["name"] == "search"],
        "retrieved_titles": seen,
        "llm_calls": sum(isinstance(m, AIMessage) for m in messages),
        "stopped_at_limit": stopped,
        "latency_s": time.perf_counter() - start,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?")
    p.add_argument("--llm", required=True, help="hf:<path or hub id>")
    p.add_argument("--index", required=True, help="dense index dir")
    p.add_argument("--retriever", default="hybrid_rerank")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--eval", action="store_true", help="run on the benchmark sample instead of one question")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--name", default="agent")
    p.add_argument("--out", default="results/agent")
    args = p.parse_args(argv)

    import anyio

    from configs.training_config import RAGConfig
    from src.llm import make_llm
    from src.mcp_server import create_server, lazy_pipeline
    from src.rag_pipeline import load_retriever
    from src.telemetry import setup_tracing

    setup_tracing("multihop-rag-agent")
    config = RAGConfig(llm=args.llm, retriever=args.retriever, index_dir=args.index)
    retriever = load_retriever(config)
    server = create_server(retriever, config, lazy_pipeline(retriever, config), llm_name=args.llm)
    llm = make_llm(args.llm, max_batch_size=args.concurrency)
    model = LocalToolChatModel(llm=llm)

    async def go():
        from mcp.client import Client

        async with Client(server) as client:
            agent = build_agent(model, await load_mcp_tools(client))
            if not args.eval:
                result = await run_agent(agent, args.question, args.max_steps)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return
            await evaluate_agent(agent, args)

    anyio.run(go)
    llm.close()


async def evaluate_agent(agent, args):
    import anyio
    from tqdm import tqdm

    from src.hotpotqa import load_hotpotqa

    examples, _ = load_hotpotqa(args.n, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.name}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [e for e in examples if e.id not in done]
    limiter = anyio.Semaphore(args.concurrency)
    bar = tqdm(total=len(todo), desc=args.name)
    start = time.perf_counter()

    with open(path, "a", encoding="utf-8") as f:
        async def one(ex):
            async with limiter:
                r = await run_agent(agent, ex.question, args.max_steps)
            f.write(json.dumps({"id": ex.id, "question": ex.question, "gold": ex.answer, "pred": r["answer"],
                                "type": ex.type, "level": ex.level, "gold_titles": ex.gold_titles,
                                **{k: v for k, v in r.items() if k != "answer"}}, ensure_ascii=False) + "\n")
            f.flush()
            bar.update(1)

        async with anyio.create_task_group() as tg:
            for ex in todo:
                tg.start_soon(one, ex)
    bar.close()
    print(f"{len(todo)} questions in {time.perf_counter() - start:.0f}s -> {path}")


if __name__ == "__main__":
    main()
