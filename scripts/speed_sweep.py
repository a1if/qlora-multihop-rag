"""
scripts/speed_sweep.py
Measure fine-tuning throughput for several settings and pick the fastest that fits.

Each variant runs `python -m src.finetune` for a fixed number of steps in its
own process (so one out-of-memory failure doesn't stop the sweep), then the
non-padding tokens/s and peak VRAM from metrics.json are collected into a
table. Runs go to $QMR_RUNS_DIR/speed_<name>.

Usage:
    python scripts/speed_sweep.py                  # default variants, 30 steps each
    python scripts/speed_sweep.py --steps 50 --only lora_gc lora_nogc_bs4
"""

import argparse
import json
import os
import subprocess
import sys

COMMON = ["max_steps={steps}", "merge_after_training=false", "max_val_examples=0", "save_steps=100000",
          "logging_steps=5"]

# name -> overrides; effective batch is 16 sequences in every variant
VARIANTS = {
    # answer-span logits off vs on, at a batch size where full logits still fit
    "lora_gc_bs4_fulllogits": ["method=lora", "gradient_checkpointing=true", "per_device_train_batch_size=4",
                               "gradient_accumulation_steps=4", "answer_span_logits=false"],
    "lora_gc_bs4":    ["method=lora", "gradient_checkpointing=true", "per_device_train_batch_size=4",
                       "gradient_accumulation_steps=4"],
    "qlora_gc":       ["method=qlora", "gradient_checkpointing=true", "per_device_train_batch_size=8",
                       "gradient_accumulation_steps=2"],
    "lora_gc":        ["method=lora", "gradient_checkpointing=true", "per_device_train_batch_size=8",
                       "gradient_accumulation_steps=2"],
    "lora_gc_bs16":   ["method=lora", "gradient_checkpointing=true", "per_device_train_batch_size=16",
                       "gradient_accumulation_steps=1"],
    "lora_nogc_bs4":  ["method=lora", "gradient_checkpointing=false", "per_device_train_batch_size=4",
                       "gradient_accumulation_steps=4"],
    "lora_nogc_bs8":  ["method=lora", "gradient_checkpointing=false", "per_device_train_batch_size=8",
                       "gradient_accumulation_steps=2"],
}


def run_variant(name: str, overrides: list[str], steps: int, runs_dir: str) -> dict:
    run_name = f"speed_{name}"
    sets = [o.format(steps=steps) for o in COMMON] + overrides + [f"run_name={run_name}"]
    cmd = [sys.executable, "-m", "src.finetune"] + [arg for o in sets for arg in ("--set", o)]
    log_path = os.path.join(runs_dir, f"{run_name}.log")
    os.makedirs(runs_dir, exist_ok=True)
    print(f"\n=== {name}: {' '.join(overrides)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

    metrics_path = os.path.join(runs_dir, run_name, "metrics.json")
    if proc.returncode != 0 or not os.path.exists(metrics_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            tail = f.read()[-3000:]
        reason = "out of memory" if "out of memory" in tail.lower() else f"exit code {proc.returncode}"
        print(f"    failed: {reason} (see {log_path})", flush=True)
        return {"variant": name, "status": reason}
    with open(metrics_path) as f:
        m = json.load(f)
    print(f"    {m['tokens_per_s']:.0f} tok/s · peak {m['peak_vram_gb']:.1f} GB · {m['train_runtime_s']:.0f} s", flush=True)
    return {"variant": name, "status": "ok", **m}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--only", nargs="*", help="subset of variant names")
    p.add_argument("--out", default="results/speed_sweep.md")
    args = p.parse_args()

    runs_dir = os.environ.get("QMR_RUNS_DIR", "runs")
    names = args.only or list(VARIANTS)
    rows = [run_variant(n, VARIANTS[n], args.steps, runs_dir) for n in names]

    ok = [r for r in rows if r["status"] == "ok"]
    ref = next((r for r in ok if r["variant"] == "lora_gc_bs4_fulllogits"), None)
    lines = [f"Fine-tuning throughput · Qwen3-1.7B · {args.steps} steps · effective batch 16 · max_len 1024", "",
             "| Variant | Status | Tokens/s | Speed vs full-logits baseline | Peak VRAM (GB) |", "|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| `{r['variant']}` | {r['status']} | – | – | – |")
            continue
        speed = f"{r['tokens_per_s'] / ref['tokens_per_s']:.2f}×" if ref else "–"
        lines.append(f"| `{r['variant']}` | ok | {r['tokens_per_s']:.0f} | {speed} | {r['peak_vram_gb']:.1f} |")
    table = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)
    if ok:
        best = max(ok, key=lambda r: r["tokens_per_s"])
        print(f"\nFastest: {best['variant']}")


if __name__ == "__main__":
    main()
