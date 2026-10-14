"""
evals/record.py
Runs the golden questions through the MCP server's `answer` tool (real model,
real retrieval) and records the outputs, so CI can check them without a GPU.

The pipeline answers over the benchmark's full pooled corpus (thousands of
paragraphs), not just the eval corpus, so retrieval is as hard as in the benchmark.

Usage (GPU):
    python -m evals.record --llm hf:D:/qmr/runs/sft_full/merged --retriever hybrid_rerank \\
        --index results/full/index_n500_seed0 --name sft_full_hybrid_rerank
"""

import argparse
import json
import os

import anyio

from configs.training_config import RAGConfig
from evals.checks import HERE, evaluate, golden


async def record(server, questions: list[dict], concurrency: int = 8) -> list[dict]:
    from mcp.client import Client

    results: dict[str, dict] = {}
    limiter = anyio.Semaphore(concurrency)

    async with Client(server) as client:
        async def one(q):
            async with limiter:
                res = await client.call_tool("answer", {"question": q["question"], "mode": "iterative"})
                if res.is_error:
                    raise RuntimeError(f"{q['id']}: {res.content[0].text}")
                results[q["id"]] = {"id": q["id"], "question": q["question"], **res.structured_content}

        async with anyio.create_task_group() as tg:
            for q in questions:
                tg.start_soon(one, q)
    return [results[q["id"]] for q in questions]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", required=True)
    p.add_argument("--retriever", default="hybrid_rerank")
    p.add_argument("--index", required=True, help="dense index dir (from the benchmark run)")
    p.add_argument("--name", required=True)
    args = p.parse_args(argv)

    from src.mcp_server import create_server, lazy_pipeline
    from src.rag_pipeline import load_retriever

    config = RAGConfig(llm=args.llm, retriever=args.retriever, index_dir=args.index)
    retriever = load_retriever(config)
    server = create_server(retriever, config, lazy_pipeline(retriever, config), llm_name=args.llm)
    records = anyio.run(record, server, golden())

    out_dir = os.path.join(HERE, "recorded")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.name}.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(evaluate(records, golden()), indent=2))


if __name__ == "__main__":
    main()
