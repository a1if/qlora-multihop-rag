"""
evals/build_golden.py
Builds the small regression eval set committed to the repo.

Takes the first N questions of the benchmark's seeded HotpotQA dev sample (so
they are part of the 500-question benchmark) and writes:

    evals/golden.jsonl   id, question, answer, type, gold_titles
    evals/corpus.jsonl   the 10 context paragraphs of each question (2 gold + 8 distractors)

HotpotQA is distributed under CC BY-SA 4.0 (Yang et al., 2018).

Usage:
    python -m evals.build_golden --n 25
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--seed", type=int, default=0, help="same seed as the benchmark sample")
    args = p.parse_args(argv)

    from datasets import load_dataset

    rows = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").shuffle(seed=args.seed)
    rows = rows.select(range(args.n))

    corpus, golden = {}, []
    for row in rows:
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"], strict=True):
            corpus.setdefault(title, "".join(sentences).strip())
        golden.append({
            "id": row["id"], "question": row["question"], "answer": row["answer"], "type": row["type"],
            "gold_titles": sorted(set(row["supporting_facts"]["title"])),
        })

    with open(os.path.join(HERE, "golden.jsonl"), "w", encoding="utf-8") as f:
        for g in golden:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for i, (title, text) in enumerate(corpus.items()):
            f.write(json.dumps({"id": i, "title": title, "text": text}, ensure_ascii=False) + "\n")
    print(f"{len(golden)} questions, {len(corpus)} paragraphs")


if __name__ == "__main__":
    main()
