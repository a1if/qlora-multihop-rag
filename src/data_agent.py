"""
src/data_agent.py
Tool-use fine-tuning data for the LangGraph agent (src/agent.py).

A 1.7B model left to plan its own tool use searches once and stops (89 of 100
benchmark questions), so it misses the second fact multi-hop questions need.
This builds HotpotQA *train* traces that show the behaviour we want:

    search(question)          -> results contain the hop-1 paragraph
    search(<bridge entity>)   -> results contain the hop-2 paragraph
    Answer: <answer>

with variants that teach when to stop early (both facts already found after
one search) and when to abstain (the second search finds nothing useful).

Every training step is rendered exactly as the agent sees it at inference: the
real MCP tools (built in-process from src/mcp_server.py), the agent's system
prompt, the same LocalToolChatModel chat-template rendering and the same
tool-result formatting. Loss is on the assistant turn only.

Usage:
    python -m src.data_agent --n 2500 --out data/agent
"""

import argparse
import json
import os
import random

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent import FIXED_ARGS, SYSTEM_PROMPT, LocalToolChatModel, format_tool_result
from src.data_sft import SFTExample, order_gold, paragraphs, stats, write_jsonl
from src.text_utils import ABSTAIN

PREVIEW_CHARS = 700          # = src.mcp_server.PREVIEW_CHARS
K = FIXED_ARGS["search"]["k"]
MAX_SEARCHES = 4


def tool_call_text(name: str, args: dict) -> str:
    """Exactly how Qwen3 emits a tool call (and how its template renders one in history)."""
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}\n</tool_call>'


def search_result(titles: list[str], paras: dict, rng: random.Random) -> dict:
    return {"results": [{"id": rng.randint(0, 4999), "title": t, "text": paras[t][:PREVIEW_CHARS],
                         "score": round(rng.uniform(0.1, 9.0), 4)} for t in titles]}


def place(first: str | None, others: list[str], rng: random.Random, top_prob: float = 0.7) -> list[str]:
    """Result list of length K with `first` usually ranked first, like the reranker does."""
    titles = list(others)[:K - (first is not None)]
    if first is not None:
        titles.insert(0 if rng.random() < top_prob else rng.randint(0, len(titles)), first)
    return titles


def build_trace(row, rng: random.Random, early_stop_ratio: float, abstain_ratio: float):
    """List of (messages-so-far, assistant reply) steps for one question, or None if unusable."""
    paras = paragraphs(row)
    gold = sorted(set(row["supporting_facts"]["title"]) & set(paras))
    distractors = [t for t in paras if t not in gold]
    if len(gold) != 2 or len(distractors) < 2 * K - 2:
        return None
    rng.shuffle(distractors)
    hop1, hop2 = order_gold(row["question"], gold)
    question = row["question"]

    messages = [SystemMessage(SYSTEM_PROMPT.format(max_searches=MAX_SEARCHES)), HumanMessage(question)]
    steps = []

    def act(reply: str, call: dict | None = None, result_titles: list[str] | None = None):
        steps.append((list(messages), reply))
        if call is None:
            return
        call_id = f"call_{rng.getrandbits(48):012x}"
        messages.append(AIMessage("", tool_calls=[{"name": "search", "args": call, "id": call_id}]))
        data = search_result(result_titles, paras, rng)
        messages.append(ToolMessage(format_tool_result("search", data), tool_call_id=call_id))

    roll = rng.random()
    if roll < early_stop_ratio:
        # Both facts come back from the first search: answer without a second search.
        act(tool_call_text("search", {"query": question}), {"query": question},
            place(hop1, [hop2] + distractors[:K - 2], rng))
        act(f"Answer: {row['answer']}")
    elif roll < early_stop_ratio + abstain_ratio:
        # The bridge search finds nothing relevant: abstain.
        act(tool_call_text("search", {"query": question}), {"query": question},
            place(hop1, distractors[:K - 1], rng))
        act(tool_call_text("search", {"query": hop2}), {"query": hop2}, place(None, distractors[K - 1:2 * K - 1], rng))
        act(f"Answer: {ABSTAIN}")
    else:
        act(tool_call_text("search", {"query": question}), {"query": question},
            place(hop1, distractors[:K - 1], rng))
        act(tool_call_text("search", {"query": hop2}), {"query": hop2},
            place(hop2, distractors[K - 1:2 * K - 2], rng, top_prob=0.8))
        act(f"Answer: {row['answer']}")
    return steps


def build_examples(rows, model: LocalToolChatModel, early_stop_ratio=0.15, abstain_ratio=0.1, seed=0):
    rng = random.Random(seed)
    examples = []
    for row in rows:
        trace = build_trace(row, rng, early_stop_ratio, abstain_ratio)
        if trace is None:
            continue
        for i, (history, reply) in enumerate(trace):
            examples.append(SFTExample(f"{row['id']}#{i}", "agent", model.render(history), reply, prerendered=True))
    return examples


async def agent_tool_schemas() -> list:
    """The tools exactly as the agent sees them: real MCP server -> load_mcp_tools (k hidden)."""
    from mcp.client import Client

    from configs.training_config import RAGConfig
    from evals.checks import corpus
    from src.agent import load_mcp_tools
    from src.mcp_server import create_server
    from src.retriever import BM25Retriever

    server = create_server(BM25Retriever(corpus()), RAGConfig(), lambda: None)
    async with Client(server) as client:
        return await load_mcp_tools(client)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=2500, help="HotpotQA train questions (traces)")
    p.add_argument("--val-n", type=int, default=100)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    p.add_argument("--seed", type=int, default=1, help="differs from data_sft's seed, so different questions")
    p.add_argument("--early-stop-ratio", type=float, default=0.15)
    p.add_argument("--abstain-ratio", type=float, default=0.10,
                   help="0.10 = final agent. 0.03 cut over-refusal but the agent then never stopped searching "
                        "(looped to the step limit on 100/100 questions); see REPORT.md section 5")
    p.add_argument("--out", default="data/agent")
    args = p.parse_args(argv)

    import anyio
    from datasets import load_dataset
    from transformers import AutoTokenizer

    class TokenizerOnly:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    model = LocalToolChatModel(llm=TokenizerOnly()).bind_tools(anyio.run(agent_tool_schemas))
    rows = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train").shuffle(seed=args.seed)
    rows = rows.select(range(args.n + args.val_n))
    ratios = dict(early_stop_ratio=args.early_stop_ratio, abstain_ratio=args.abstain_ratio)
    splits = {"train": build_examples(rows.select(range(args.val_n, args.n + args.val_n)), model, seed=args.seed, **ratios),
              "val": build_examples(rows.select(range(args.val_n)), model, seed=args.seed + 1, **ratios)}

    os.makedirs(args.out, exist_ok=True)
    summary = {"args": vars(args)}
    for name, examples in splits.items():
        write_jsonl(os.path.join(args.out, f"{name}.jsonl"), examples)
        summary[name] = stats(examples)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
