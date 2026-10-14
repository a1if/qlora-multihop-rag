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
import contextvars
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
    "Most questions need two facts from two different articles. Work step by step:\n"
    "1. Search for the question.\n"
    "2. Read the results. If they name a person, work, place or organisation that the question "
    "depends on, search for that name next (e.g. after finding who voices a character, search for "
    "that actor).\n"
    "3. Repeat until the passages contain the answer, using at most {max_searches} searches.\n"
    "Use `get_passage` only if a preview is cut off before the fact you need.\n"
    "When the passages contain the answer, reply with exactly one line: 'Answer: <short answer>' "
    "(an entity, date, number, or yes/no). Only after at least two searches without finding it, reply "
    "'Answer: insufficient evidence'."
)

# Arguments the agent doesn't get to choose: a 1.7B model picks k=1 and starves itself of evidence.
FIXED_ARGS = {"search": {"k": 5}}
MAX_SEARCHES = 4

# Guardrails, enforced in the tool layer per run: a small model can loop on the same failed search
# (agent v2 searched "Ted (film)" until the step limit on all 100 questions).
_RUN = contextvars.ContextVar("agent_run", default=None)
REPEATED_SEARCH = ("You already searched for '{query}'. Do not search it again. Answer from the passages above "
                   "with 'Answer: <short answer>', or reply 'Answer: insufficient evidence'.")
FINAL_ANSWER = ("Stop searching. Using only the passages above, reply with exactly one line: "
                "'Answer: <short answer>', or 'Answer: insufficient evidence' if they don't contain it.")
SEARCH_LIMIT = ("Search limit reached ({limit} searches). Answer now from the passages above with "
                "'Answer: <short answer>', or reply 'Answer: insufficient evidence'.")


def _guard(name: str, args: dict) -> str | None:
    """Message to return instead of running the tool, or None to run it."""
    run = _RUN.get()
    if run is None or name != "search":
        return None
    query = " ".join(str(args.get("query", "")).lower().split())
    if query in run["queries"]:
        run["guardrail_hits"] += 1
        return REPEATED_SEARCH.format(query=args.get("query", ""))
    if len(run["queries"]) >= run["max_searches"]:
        run["guardrail_hits"] += 1
        return SEARCH_LIMIT.format(limit=run["max_searches"])
    run["queries"].append(query)
    return None

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
    reasoning: bool = False          # Qwen3 thinking mode: better tool planning, ~3-5x more tokens
    tool_schemas: list[dict] = []

    @property
    def _llm_type(self) -> str:
        return "local-hf-tool-chat"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tool_schemas": [convert_to_openai_tool(t) for t in tools]})

    def render(self, messages: list[BaseMessage]) -> str:
        return self.llm.tokenizer.apply_chat_template(
            to_chat_dicts(messages), tools=self.tool_schemas or None, tokenize=False,
            add_generation_prompt=True, enable_thinking=self.reasoning,
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


def _without(schema: dict, fixed: dict) -> dict:
    """Input schema with the fixed arguments removed, so the model never sees them."""
    props = {k: v for k, v in schema.get("properties", {}).items() if k not in fixed}
    required = [r for r in schema.get("required", []) if r not in fixed]
    return {**schema, "properties": props, "required": required}


async def load_mcp_tools(client, allowed: tuple[str, ...] = ("search", "get_passage"),
                         fixed_args: dict | None = None) -> list[StructuredTool]:
    """
    Wrap an open MCP client's tools as LangChain tools (content for the model, artifact = structured result).
    fixed_args pins some arguments (default: search k=5) and hides them from the model.
    """
    from opentelemetry.propagate import inject

    fixed_args = FIXED_ARGS if fixed_args is None else fixed_args
    tools = []
    for t in (await client.list_tools()).tools:
        if t.name not in allowed:
            continue
        fixed = fixed_args.get(t.name, {})

        async def call(_name=t.name, _fixed=fixed, **kwargs):
            blocked = _guard(_name, kwargs)
            if blocked is not None:
                return blocked, {"guardrail": True}
            meta: dict = {}
            inject(meta)                       # continue the agent's trace inside the server
            result = await client.call_tool(_name, {**kwargs, **_fixed}, meta=meta or None)
            if result.is_error:
                return f"ERROR: {result.content[0].text}", {"error": True}
            data = result.structured_content or {}
            return format_tool_result(_name, data), data

        tools.append(StructuredTool(name=t.name, description=t.description or t.name,
                                    args_schema=_without(t.input_schema, fixed),
                                    coroutine=call, response_format="content_and_artifact"))
    return tools


def build_agent(chat_model: BaseChatModel, tools: list, max_searches: int = MAX_SEARCHES):
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(chat_model, tools, prompt=SYSTEM_PROMPT.format(max_searches=max_searches))


# ─── Running ──────────────────────────────────────────────────────────────────

def _history_for_final(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Conversation up to the last completed tool result, dropping LangGraph's out-of-steps
    message and any trailing tool call that never got a response."""
    keep = list(messages)
    while keep and isinstance(keep[-1], AIMessage):
        keep.pop()
    return keep


async def force_final_answer(chat_model, messages: list[BaseMessage]) -> str:
    """One tool-free call asking for an answer from the evidence gathered so far."""
    import anyio

    model = chat_model.model_copy(update={"tool_schemas": []})
    # Prefill "Answer:" so the reply can only be an answer: a model tuned hard on tool use
    # otherwise emits another tool call even when no tools are offered.
    prompt = model.render(_history_for_final(messages) + [HumanMessage(FINAL_ANSWER)]) + "Answer:"
    text = await anyio.to_thread.run_sync(model.llm.complete, prompt, 32)
    return "Answer:" + strip_reasoning(text).split("\n")[0]


async def run_agent(agent, question: str, max_steps: int = 6, max_searches: int = MAX_SEARCHES,
                    finalizer=None) -> dict:
    """
    One question -> answer, tool calls made, passages seen, latency.
    finalizer: the agent's chat model; if the run ends without an answer (step limit, ignored
    guardrails), it gets one tool-free call to answer from what was found ("forced" answer).
    """
    from langgraph.errors import GraphRecursionError

    start = time.perf_counter()
    run = {"queries": [], "max_searches": max_searches, "guardrail_hits": 0}
    token = _RUN.set(run)
    with span("agent.run", agent__question_chars=len(question)) as s:
        messages, stopped = [], False
        try:
            # stream states so a run cut off by the step limit still reports what it did
            async for state in agent.astream({"messages": [HumanMessage(question)]}, stream_mode="values",
                                             config={"recursion_limit": 2 * max_steps + 1}):
                messages = state["messages"]
        except GraphRecursionError:
            stopped = True
        finally:
            _RUN.reset(token)
        calls = [c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
        seen = []
        for m in messages:
            if isinstance(m, ToolMessage) and isinstance(m.artifact, dict):
                passages = m.artifact.get("results") or ([m.artifact] if "title" in m.artifact else [])
                seen += [p["title"] for p in passages if p.get("title") not in seen]
        final = next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), "")
        if final.startswith("Sorry, need more steps"):    # create_react_agent's out-of-steps message
            final, stopped = "", True
        if stopped:
            final = ""
        forced = False
        if not final and finalizer is not None and messages:
            final, forced = await force_final_answer(finalizer, messages), True
        s.set_attributes({"agent.tool_calls": len(calls), "agent.stopped_at_limit": stopped,
                          "agent.guardrail_hits": run["guardrail_hits"], "agent.forced_answer": forced})
    return {
        "answer": extract_answer(final) if final else "",
        "raw": final,
        "tool_calls": [{"name": c["name"], "args": c["args"]} for c in calls],
        "queries": [c["args"].get("query") for c in calls if c["name"] == "search"],
        "retrieved_titles": seen,
        "llm_calls": sum(isinstance(m, AIMessage) for m in messages),
        "stopped_at_limit": stopped,
        "guardrail_hits": run["guardrail_hits"],
        "forced_answer": forced,
        "latency_s": time.perf_counter() - start,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?")
    p.add_argument("--llm", required=True, help="hf:<path or hub id>")
    p.add_argument("--index", required=True, help="dense index dir")
    p.add_argument("--retriever", default="hybrid_rerank")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--reasoning", action="store_true", help="Qwen3 thinking mode before each step")
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
    model = LocalToolChatModel(llm=llm, reasoning=args.reasoning, max_new_tokens=768 if args.reasoning else 160)

    async def go():
        from mcp.client import Client

        async with Client(server) as client:
            agent = build_agent(model, await load_mcp_tools(client))
            if not args.eval:
                result = await run_agent(agent, args.question, args.max_steps, finalizer=model)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return
            await evaluate_agent(agent, args, finalizer=model)

    anyio.run(go)
    print(f"OOM batch splits: {getattr(llm, 'oom_splits', 0)}")
    llm.close()


async def evaluate_agent(agent, args, finalizer=None):
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
                try:
                    r = await run_agent(agent, ex.question, args.max_steps, finalizer=finalizer)
                except Exception as exc:          # one failed question must not abort the run
                    r = {"answer": "", "raw": "", "tool_calls": [], "queries": [], "retrieved_titles": [],
                         "llm_calls": 0, "stopped_at_limit": False, "latency_s": 0.0,
                         "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
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
