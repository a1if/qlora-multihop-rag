"""
src/eval_hotpotqa.py
Ablation benchmark on HotpotQA (distractor dev set, pooled-corpus retrieval).

Each system is "<model>:<mode>":
    model  a name bound with --model NAME=SPEC (defaults: base, ft)
    mode   closed | single | iterative | decompose (see src/rag_pipeline.py)

SPEC is any src/llm.py backend, so the same benchmark runs against a local
transformers model or a checkpoint served by vLLM / Ollama.

Reports, per system:
    EM / F1 with 95% bootstrap CIs (official HotpotQA answer normalisation)
    Support recall@k and all-support@k (are the gold paragraphs retrieved?)
    EM split by question type (bridge vs comparison)
    Throughput (questions/s, wall clock) and p50 / p95 latency under load,
    LLM calls per question, time split retrieval vs LLM
    Paired-bootstrap delta in F1 against a reference system

Predictions are appended to JSONL as they complete, so an interrupted run
resumes where it stopped.

Usage:
    # smoke test against a local Ollama model
    python -m src.eval_hotpotqa --n 20 \\
        --model base=openai:qwen3:1.7b@http://localhost:11434/v1 \\
        --systems base:closed base:single

    # full ablation: base model vs the QLoRA fine-tune, both served by vLLM
    python -m src.eval_hotpotqa --n 500 \\
        --model base=openai:base@http://localhost:8000/v1 \\
        --model ft=openai:ft@http://localhost:8001/v1 \\
        --systems base:closed base:single base:iterative base:decompose \\
                  ft:closed ft:single ft:iterative

    # re-print the table from saved predictions
    python -m src.eval_hotpotqa --n 500 --systems ... --summarize-only
"""

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields

import numpy as np

from configs.training_config import FinetuneConfig, RAGConfig
from src.metrics import (
    all_support_found, bootstrap_ci, exact_match, f1_score, paired_bootstrap_delta, support_recall,
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    rag = RAGConfig()
    ft = FinetuneConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=500, help="number of dev questions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", action="append", default=[], metavar="NAME=SPEC",
                   help="bind a model name to an LLM spec (repeatable)")
    p.add_argument("--systems", nargs="+", default=["base:closed", "base:single", "base:iterative", "base:decompose"])
    p.add_argument("--reference", default="base:single", help="system to compute F1 deltas against")
    p.add_argument("--workers", type=int, default=16,
                   help="concurrent questions; for hf: models this is also the dynamic batch size")
    p.add_argument("--out", default="results/hotpotqa")
    p.add_argument("--top-k", type=int, default=rag.top_k)
    p.add_argument("--hops", type=int, default=rag.num_hops)
    p.add_argument("--prompt-style", choices=["chat", "alpaca"], default=rag.prompt_style)
    p.add_argument("--reasoning", action="store_true", help="allow <think> reasoning (much slower)")
    p.add_argument("--load-in-4bit", action="store_true", help="hf backend only")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args(argv)

    args.models = {"base": f"hf:{ft.base_model_name}", "ft": f"hf:{ft.merged_dir}"}
    for binding in args.model:
        name, eq, spec = binding.partition("=")
        if not eq or not name or not spec:
            p.error(f"--model expects NAME=SPEC, got {binding!r}")
        args.models[name] = spec
    for system in args.systems:
        model, _, mode = system.partition(":")
        if model not in args.models or not mode:
            p.error(f"Bad system {system!r}; models available: {sorted(args.models)}")
    return args


def split_system(system: str) -> tuple[str, str]:
    model, _, mode = system.partition(":")
    return model, mode


def preds_path(args, system: str) -> str:
    """Predictions are keyed by every setting that changes them, so runs never mix."""
    tag = f"n{args.n}_seed{args.seed}_k{args.top_k}_h{args.hops}_{args.prompt_style}"
    tag += "_think" if args.reasoning else ""
    tag += "_4bit" if args.load_in_4bit else ""
    return os.path.join(args.out, "preds", tag, system.replace(":", "__") + ".jsonl")


def load_preds(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


# ─── Running Systems ──────────────────────────────────────────────────────────

def build_retriever(docs, config, out_dir, n, seed):
    from sentence_transformers import SentenceTransformer
    from src.retriever import DenseRetriever

    embedder = SentenceTransformer(config.embedding_model)
    index_dir = os.path.join(out_dir, f"index_n{n}_seed{seed}")
    if DenseRetriever.exists(index_dir):
        return DenseRetriever.load(index_dir, embedder)
    print(f"Embedding {len(docs)} HotpotQA paragraphs...")
    retriever = DenseRetriever.build(docs, embedder)
    retriever.save(index_dir)
    return retriever


def prediction_record(ex, result) -> dict:
    return {
        "id": ex.id,
        "question": ex.question,
        "gold": ex.answer,
        "pred": result.answer,
        "raw": result.raw_output,
        "type": ex.type,
        "level": ex.level,
        "queries": result.queries,
        "retrieved_titles": [p["title"] for p in result.passages],
        "gold_titles": ex.gold_titles,
        "latency_s": result.latency_s,
        **result.timings,
    }


def meta_path(path: str) -> str:
    return path.replace(".jsonl", ".meta.json")


def load_segments(path: str) -> list[dict]:
    """Wall-clock segments for a system (one per run; resumed runs add more)."""
    if not os.path.exists(meta_path(path)):
        return []
    with open(meta_path(path)) as f:
        return json.load(f).get("segments", [])


def run_system(pipeline, system, examples, path, workers):
    from tqdm import tqdm

    mode = split_system(system)[1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    start = time.perf_counter()
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(pipeline.answer, ex.question, mode): ex for ex in examples}
        for future in tqdm(as_completed(futures), total=len(futures), desc=system):
            f.write(json.dumps(prediction_record(futures[future], future.result()), ensure_ascii=False) + "\n")
            f.flush()
    segments = load_segments(path) + [
        {"questions": len(examples), "wall_s": time.perf_counter() - start, "workers": workers}]
    with open(meta_path(path), "w") as f:
        json.dump({"segments": segments}, f, indent=2)


def run_systems(args, config, examples, docs):
    from src.llm import make_llm
    from src.rag_pipeline import RAGPipeline

    retriever = build_retriever(docs, config, args.out, args.n, args.seed)

    by_model = defaultdict(list)
    for system in args.systems:
        by_model[split_system(system)[0]].append(system)

    for model_name, systems in by_model.items():
        todo = {s: [e for e in examples if e.id not in load_preds(preds_path(args, s))] for s in systems}
        if not any(todo.values()):
            continue

        spec = args.models[model_name]
        workers = args.workers
        print(f"\nModel {model_name!r} -> {spec} (workers={workers})")
        # hf: concurrent calls from the worker threads are batched on the GPU
        llm = make_llm(spec, config.prompt_style, config.reasoning, config.load_in_4bit, max_batch_size=workers)
        pipeline = RAGPipeline(llm, retriever, config)
        try:
            for system in systems:
                if todo[system]:
                    run_system(pipeline, system, todo[system], preds_path(args, system), workers)
        finally:
            llm.close()


# ─── Summary ──────────────────────────────────────────────────────────────────

def summarize(args, examples) -> list[dict]:
    ids = [e.id for e in examples]
    scores, rows = {}, []

    for system in args.systems:
        preds = load_preds(preds_path(args, system))
        missing = [i for i in ids if i not in preds]
        if missing:
            print(f"[warn] {system}: {len(missing)} of {len(ids)} questions have no prediction; skipped")
            continue
        recs = [preds[i] for i in ids]
        em = [exact_match(r["pred"], r["gold"]) for r in recs]
        f1 = [f1_score(r["pred"], r["gold"]) for r in recs]
        scores[system] = {"em": em, "f1": f1}

        row = {"system": system, "llm": args.models[split_system(system)[0]]}
        for name, vals in (("EM", em), ("F1", f1)):
            mean, lo, hi = bootstrap_ci(vals, seed=args.seed)
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = mean, lo, hi
        for qtype in ("bridge", "comparison"):
            vals = [e for e, r in zip(em, recs, strict=True) if r["type"] == qtype]
            row[f"EM_{qtype}"] = float(np.mean(vals)) if vals else float("nan")
        if split_system(system)[1] != "closed":
            row["SupportRecall@k"] = float(np.mean([support_recall(r["retrieved_titles"], r["gold_titles"]) for r in recs]))
            row["AllSupport@k"] = float(np.mean([all_support_found(r["retrieved_titles"], r["gold_titles"]) for r in recs]))
        latencies = [r["latency_s"] for r in recs]
        row["latency_p50_s"] = float(np.percentile(latencies, 50))
        row["latency_p95_s"] = float(np.percentile(latencies, 95))
        row["llm_calls"] = float(np.mean([r.get("llm_calls", 0) for r in recs]))
        row["retrieval_s"] = float(np.mean([r.get("retrieval_s", 0.0) for r in recs]))
        row["llm_s"] = float(np.mean([r.get("llm_s", 0.0) for r in recs]))
        segments = load_segments(preds_path(args, system))
        wall = sum(seg["wall_s"] for seg in segments)
        row["throughput_qps"] = sum(seg["questions"] for seg in segments) / wall if wall else float("nan")
        row["workers"] = max((seg["workers"] for seg in segments), default=None)
        rows.append(row)

    ref = scores.get(args.reference)
    for row in rows:
        if ref is not None and row["system"] != args.reference:
            d, lo, hi = paired_bootstrap_delta(scores[row["system"]]["f1"], ref["f1"], seed=args.seed)
            row["dF1_vs_ref"], row["dF1_lo"], row["dF1_hi"] = d, lo, hi

    write_summary(args, rows, len(ids))
    return rows


def _pct(row, key):
    return f"{100 * row[key]:.1f}" if key in row and row[key] == row[key] else "–"


def write_summary(args, rows, n):
    if not rows:
        print("No complete systems to summarise.")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(args.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"HotpotQA distractor dev · n={n} · seed={args.seed} · k={args.top_k} · hops={args.hops} · "
        f"prompt={args.prompt_style} · reasoning={args.reasoning}",
        "",
        f"| System | EM | F1 (95% CI) | EM bridge | EM comp. | Support R@{args.top_k} | All-support@{args.top_k} "
        f"| ΔF1 vs {args.reference} (95% CI) | Throughput (q/s) | p50 / p95 latency under load (s) | LLM calls | LLM share |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        delta = (f"{100 * r['dF1_vs_ref']:+.1f} [{100 * r['dF1_lo']:+.1f}, {100 * r['dF1_hi']:+.1f}]"
                 if "dF1_vs_ref" in r else "–")
        qps = f"{r['throughput_qps']:.2f}" if r["throughput_qps"] == r["throughput_qps"] else "–"
        total = r["llm_s"] + r["retrieval_s"]
        llm_share = f"{100 * r['llm_s'] / total:.0f}%" if total else "–"
        lines.append(
            f"| `{r['system']}` | {_pct(r, 'EM')} | {_pct(r, 'F1')} [{_pct(r, 'F1_lo')}, {_pct(r, 'F1_hi')}] "
            f"| {_pct(r, 'EM_bridge')} | {_pct(r, 'EM_comparison')} | {_pct(r, 'SupportRecall@k')} "
            f"| {_pct(r, 'AllSupport@k')} | {delta} | {qps} | {r['latency_p50_s']:.2f} / {r['latency_p95_s']:.2f} "
            f"| {r['llm_calls']:.1f} | {llm_share} |"
        )
    table = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main(argv=None, examples=None, docs=None):
    args = parse_args(argv)
    config = RAGConfig(top_k=args.top_k, num_hops=args.hops, prompt_style=args.prompt_style,
                       reasoning=args.reasoning, load_in_4bit=args.load_in_4bit)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        run_config = {k: v for k, v in vars(args).items() if k != "model"}
        json.dump({**run_config, "rag_config": {fl.name: getattr(config, fl.name) for fl in fields(config)}},
                  f, indent=2)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if examples is None:
        from src.hotpotqa import load_hotpotqa
        examples, docs = load_hotpotqa(args.n, seed=args.seed)
    print(f"{len(examples)} questions · {len(docs)} pooled paragraphs")

    if not args.summarize_only:
        run_systems(args, config, examples, docs)
    return summarize(args, examples)


if __name__ == "__main__":
    main()
