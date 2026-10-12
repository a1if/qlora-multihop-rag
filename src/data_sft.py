"""
src/data_sft.py
Builds supervised fine-tuning data from the HotpotQA *train* split.

One LoRA adapter is trained on two tasks, each matching a model call in
src/rag_pipeline.py (prompts are built with the same functions):

    reader   question + k passages (2 gold + distractors, shuffled)
             → "Answer: <answer>\nSources: [i] [j]"
             A fraction of examples drop a gold passage and target
             "Answer: insufficient evidence" (abstention).

    query    question + passages found by the first search (the hop-1 gold
             paragraph + distractors) → the title of the hop-2 gold paragraph,
             i.e. the bridge entity the second search should look for.

HotpotQA's own distractors are hard negatives (retrieved by TF-IDF for each
question), so the reader learns to ignore plausible but irrelevant passages.

The benchmark uses the *validation* split, which shares no questions with train.

Usage:
    python -m src.data_sft --n 20000 --out data/sft
    python -m src.data_sft --n 2000 --abstain-ratio 0.0 --out data/sft_small
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass

from configs.training_config import RAGConfig
from src.rag_pipeline import followup_prompt, reader_prompt
from src.text_utils import ABSTAIN, format_target


@dataclass
class SFTExample:
    id: str
    task: str           # reader | abstain | query
    instruction: str    # user turn, rendered with the chat template at training time
    response: str       # assistant turn (the only tokens that get loss)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def paragraphs(row: dict) -> dict[str, str]:
    """title → paragraph text for one HotpotQA row."""
    ctx = row["context"]
    return {t: "".join(s).strip() for t, s in zip(ctx["title"], ctx["sentences"], strict=True)}


def order_gold(question: str, gold_titles: list[str]) -> tuple[str, str]:
    """
    Split the two gold paragraphs into (hop-1, hop-2).
    Hop 1 is the one the question names directly (title in the question, else
    the larger word overlap); hop 2 is the bridge the second search must find.
    """
    q = question.lower()
    named = [t for t in gold_titles if t.lower() in q]
    if len(named) == 1:
        first = named[0]
    else:
        first = max(gold_titles, key=lambda t: (len(_tokens(t) & _tokens(question)), t))
    second = next(t for t in gold_titles if t != first)
    return first, second


# ─── Example Builders ─────────────────────────────────────────────────────────

def reader_example(row, paras, gold, k, max_chars, rng, abstain: bool) -> SFTExample:
    distractors = [t for t in paras if t not in gold]
    if abstain:
        # Keep at most one gold paragraph so the answer is not fully supported.
        kept = rng.sample(gold, k=rng.choice([0, 1]))
        titles = kept + rng.sample(distractors, k=min(k - len(kept), len(distractors)))
    else:
        titles = list(gold) + rng.sample(distractors, k=min(k - len(gold), len(distractors)))
    rng.shuffle(titles)

    passages = [{"title": t, "text": paras[t]} for t in titles]
    if abstain:
        response = format_target(ABSTAIN, [])
    else:
        response = format_target(row["answer"], sorted(i for i, t in enumerate(titles, 1) if t in gold))
    return SFTExample(row["id"], "abstain" if abstain else "reader",
                      reader_prompt(row["question"], passages, max_chars), response)


def query_example(row, paras, gold, max_chars, rng, context: int = 3) -> SFTExample:
    first, second = order_gold(row["question"], gold)
    distractors = [t for t in paras if t not in gold]
    # The first search's results: the hop-1 paragraph, usually on top, plus distractors.
    titles = rng.sample(distractors, k=min(context - 1, len(distractors)))
    titles.insert(0 if rng.random() < 0.7 else rng.randint(0, len(titles)), first)
    passages = [{"title": t, "text": paras[t]} for t in titles]
    return SFTExample(row["id"], "query", followup_prompt(row["question"], passages, max_chars), second)


def build_examples(rows, k: int, max_chars: int, abstain_ratio: float, query_ratio: float,
                   seed: int = 0) -> list[SFTExample]:
    rng = random.Random(seed)
    examples = []
    for row in rows:
        paras = paragraphs(row)
        gold = sorted(set(row["supporting_facts"]["title"]) & set(paras))
        if len(gold) != 2 or len(paras) < k:
            continue  # malformed rows (a handful in HotpotQA)
        examples.append(reader_example(row, paras, gold, k, max_chars, rng,
                                       abstain=rng.random() < abstain_ratio))
        if rng.random() < query_ratio:
            examples.append(query_example(row, paras, gold, max_chars, rng))
    rng.shuffle(examples)
    return examples


# ─── I/O ──────────────────────────────────────────────────────────────────────

def write_jsonl(path: str, examples: list[SFTExample]):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[SFTExample]:
    with open(path, encoding="utf-8") as f:
        return [SFTExample(**json.loads(line)) for line in f]


def stats(examples: list[SFTExample]) -> dict:
    tasks = Counter(ex.task for ex in examples)
    return {
        "examples": len(examples),
        "tasks": dict(tasks),
        "avg_instruction_chars": round(sum(len(e.instruction) for e in examples) / max(len(examples), 1)),
        "avg_response_chars": round(sum(len(e.response) for e in examples) / max(len(examples), 1)),
    }


def main(argv=None):
    rag = RAGConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20000, help="HotpotQA train questions to use")
    p.add_argument("--val-n", type=int, default=500, help="questions held out for eval loss")
    p.add_argument("--k", type=int, default=rag.top_k, help="passages per reader example (match inference)")
    p.add_argument("--abstain-ratio", type=float, default=0.1)
    p.add_argument("--query-ratio", type=float, default=0.5, help="share of questions that also get a query example")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/sft")
    args = p.parse_args(argv)

    from datasets import load_dataset

    train = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train").shuffle(seed=args.seed)
    total = min(args.n + args.val_n, len(train))
    rows = train.select(range(total))
    kwargs = dict(k=args.k, max_chars=rag.max_passage_chars, abstain_ratio=args.abstain_ratio,
                  query_ratio=args.query_ratio)
    splits = {
        "train": build_examples(rows.select(range(args.val_n, total)), seed=args.seed, **kwargs),
        "val": build_examples(rows.select(range(args.val_n)), seed=args.seed + 1, **kwargs),
    }

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
