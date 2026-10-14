"""
scripts/export_results.py
Regenerate every result table in docs/results/ from the saved prediction files,
so the numbers in README.md and REPORT.md can be checked against raw outputs.

Inputs (local, gitignored): results/*/preds/**.jsonl, results/agent/*.jsonl,
$QMR_RUNS_DIR/*/metrics.json, results/speed_sweep.md.

Usage:
    python scripts/export_results.py
"""

import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import eval_hotpotqa  # noqa: E402
from src.metrics import bootstrap_ci, exact_match, f1_score, paired_bootstrap_delta  # noqa: E402
from src.text_utils import is_abstention  # noqa: E402

OUT = "docs/results"
RUNS = os.environ.get("QMR_RUNS_DIR", "D:/qmr/runs")
FT_10K = f"hf:{RUNS}/sft_10k/merged"
EMBED = f"{RUNS}/retriever_minilm"

# name -> (results dir, extra CLI args, systems, reference)
BENCHMARKS = {
    "1_ladder_dense": ("results/baseline", ["--model", f"ft={FT_10K}"],
                       ["base:closed", "base:single", "base:iterative", "base:decompose", "ft:single", "ft:iterative"],
                       "base:single"),
    "2_hybrid": ("results/baseline", ["--retriever", "hybrid", "--model", f"ft={FT_10K}"],
                 ["base:single", "base:iterative", "ft:iterative"], "base:single"),
    "3_hybrid_rerank": ("results/baseline", ["--retriever", "hybrid_rerank", "--model", f"ft={FT_10K}"],
                        ["base:single", "base:iterative", "ft:iterative"], "base:single"),
    "4_tuned_embedder_dense": ("results/baseline", ["--embedding-model", EMBED, "--model", f"ft={FT_10K}"],
                               ["base:single", "base:iterative", "ft:iterative"], "base:single"),
    "4_tuned_embedder_hybrid_rerank": ("results/baseline", ["--retriever", "hybrid_rerank", "--embedding-model", EMBED,
                                                            "--model", f"ft={FT_10K}"],
                                       ["base:single", "base:iterative", "ft:iterative"], "base:single"),
    "5_sft_full_30k": ("results/full", ["--retriever", "hybrid_rerank", "--model", f"ft=hf:{RUNS}/sft_full/merged"],
                       ["ft:single", "ft:iterative"], "ft:single"),
    "6_reference_qwen3_8b_4bit": ("results/full", ["--retriever", "hybrid_rerank", "--load-in-4bit",
                                                   "--model", "q8b=hf:Qwen/Qwen3-8B"],
                                  ["q8b:single", "q8b:iterative"], "q8b:single"),
}


def export_benchmarks(examples, docs):
    for name, (out, extra, systems, reference) in BENCHMARKS.items():
        argv = ["--n", "500", "--out", out, "--summarize-only", "--reference", reference, "--systems", *systems, *extra]
        eval_hotpotqa.main(argv, examples=examples, docs=docs)
        shutil.copy(os.path.join(out, "summary.md"), os.path.join(OUT, f"benchmark_{name}.md"))


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def export_agents():
    rows = {
        "Base Qwen3-1.7B agent": "results/agent/base_agent_plain.jsonl",
        "Base agent, thinking mode": "results/agent/base_agent_reasoning.jsonl",
        "Tool-use fine-tuned agent v1 (10% abstain traces, final)": "results/agent/ft_agent_plain.jsonl",
        "Agent v2 (3% abstain traces), no guardrails": "results/agent/ft_agent_v2_noguard.jsonl",
        "Agent v2 + guardrails + forced answer": "results/agent/ft_agent_v2_guard.jsonl",
    }
    pipeline = load(glob.glob("results/baseline/preds/*hybrid_rerank/ft__iterative.jsonl")[0])
    final = load(rows["Tool-use fine-tuned agent v1 (10% abstain traces, final)"])
    ids = sorted(final)
    lines = ["Agent benchmark: first 100 of the 500 HotpotQA dev questions, hybrid retrieval + rerank, k=5.", "",
             "| Agent | EM | F1 (95% CI) | Both gold seen | Searches / q | Abstain | ΔF1 vs fixed pipeline (95% CI) |",
             "|---|---|---|---|---|---|---|"]
    for name, path in {**rows, "Fine-tuned fixed 2-hop pipeline (reference)": None}.items():
        r = pipeline if path is None else load(path)
        f1 = [f1_score(r[i]["pred"], r[i]["gold"]) for i in ids]
        em = sum(exact_match(r[i]["pred"], r[i]["gold"]) for i in ids)
        mean, lo, hi = bootstrap_ci(f1)
        hit = sum(set(r[i]["gold_titles"]) <= set(r[i]["retrieved_titles"]) for i in ids)
        searches = sum(len(r[i]["queries"]) for i in ids) / len(ids)
        abstain = sum(is_abstention(r[i]["pred"]) for i in ids)
        if path is None:
            delta = "–"
        else:
            d, dlo, dhi = paired_bootstrap_delta(f1, [f1_score(pipeline[i]["pred"], pipeline[i]["gold"]) for i in ids])
            delta = f"{100 * d:+.1f} [{100 * dlo:+.1f}, {100 * dhi:+.1f}]"
        lines.append(f"| {name} | {em:.0f} | {100 * mean:.1f} [{100 * lo:.1f}, {100 * hi:.1f}] | {hit}% | "
                     f"{searches:.2f} | {abstain}% | {delta} |")
    with open(os.path.join(OUT, "agents.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_training_runs():
    lines = ["Training runs (RTX 5070, 12 GB). From each run's metrics.json.", "",
             "| Run | Examples / task | Steps | Train loss | Eval loss | Tokens/s | Peak VRAM (GB) | Time (min) |",
             "|---|---|---|---|---|---|---|---|"]
    notes = {"sft_10k": "10k reader/abstain/query (final pipeline model)", "sft_full": "29.7k reader/abstain/query",
             "agent_sft": "7,040 agent steps, 10% abstain (final agent)", "agent_sft_v2": "7,040 agent steps, 3% abstain"}
    for run, note in notes.items():
        path = os.path.join(RUNS, run, "metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        lines.append(f"| `{run}` | {note} | {m['steps']} | {m['train_loss']:.3f} | {m.get('eval_loss', float('nan')):.3f} "
                     f"| {m['tokens_per_s']:.0f} | {m['peak_vram_gb']:.1f} | {m['train_runtime_s'] / 60:.0f} |")
    retriever = os.path.join(RUNS, "retriever_minilm", "train_metrics.json")
    if os.path.exists(retriever):
        with open(retriever) as f:
            m = json.load(f)
        lines += ["", f"Retriever fine-tune (`retriever_minilm`): {m['triplets']:,} triplets, loss {m['train_loss']:.3f}, "
                      f"{m['train_runtime_s']:.0f} s."]
    with open(os.path.join(OUT, "training_runs.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_regression_eval():
    from evals.checks import HERE, evaluate, golden, load_jsonl, thresholds

    lines = ["Regression eval (CI): 25 golden questions, outputs recorded through the MCP `answer` tool.", "",
             "| Recording | Metric | Value | CI gate |", "|---|---|---|---|"]
    gates = thresholds()["recorded"]
    for path in sorted(glob.glob(os.path.join(HERE, "recorded", "*.jsonl"))):
        name = os.path.basename(path)[:-len(".jsonl")]
        metrics = evaluate(load_jsonl(path), golden())
        for metric, value in metrics.items():
            gate = next((f"{k.split('_', 1)[0]} {v}" for k, v in gates.get(name, {}).items()
                         if not k.startswith("_") and k.split("_", 1)[1] == metric), "–")
            shown = f"{value:.3f}" if isinstance(value, float) else str(value)
            lines.append(f"| `{name}` | {metric} | {shown} | {gate} |")
    with open(os.path.join(OUT, "regression_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    export_regression_eval()
    from src.hotpotqa import load_hotpotqa

    examples, docs = load_hotpotqa(500, seed=0)
    export_benchmarks(examples, docs)
    export_agents()
    export_training_runs()
    if os.path.exists("results/speed_sweep.md"):
        shutil.copy("results/speed_sweep.md", os.path.join(OUT, "training_speed_sweep.md"))
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
