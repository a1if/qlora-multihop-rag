"""
src/finetune_retriever.py
Contrastive fine-tuning of the dense retriever's embedding model on HotpotQA.

Training triplets come from the HotpotQA *train* split (the benchmark uses
dev): (question, gold supporting passage, hard negative), where the hard
negative is one of HotpotQA's own distractor paragraphs for that question
(chosen by TF-IDF similarity, so it looks relevant but is not).

Loss: MultipleNegativesRankingLoss: for each question the gold passage must
outscore its hard negative and every other passage in the batch. Batches use
NO_DUPLICATES sampling so a question's second gold passage is never treated
as a negative for the same question.

Passages are embedded as "title: text", exactly as src/retriever.py indexes them.

Usage:
    python -m src.finetune_retriever --n 20000 --out D:/qmr/runs/retriever_minilm
    python -m src.eval_hotpotqa --embedding-model D:/qmr/runs/retriever_minilm ...
"""

import argparse
import json
import os
import random

from configs.training_config import RAGConfig
from src.data_sft import paragraphs
from src.retriever import doc_text


def build_triplets(rows, seed: int = 0) -> list[dict]:
    """One triplet per gold passage: {anchor: question, positive: gold, negative: distractor}."""
    rng = random.Random(seed)
    triplets = []
    for row in rows:
        paras = paragraphs(row)
        gold = sorted(set(row["supporting_facts"]["title"]) & set(paras))
        distractors = [t for t in paras if t not in gold]
        if len(gold) != 2 or not distractors:
            continue
        for title in gold:
            negative = rng.choice(distractors)
            triplets.append({
                "anchor": row["question"],
                "positive": doc_text({"title": title, "text": paras[title]}),
                "negative": doc_text({"title": negative, "text": paras[negative]}),
            })
    rng.shuffle(triplets)
    return triplets


def train(triplets: list[dict], base_model: str, out_dir: str, epochs: float, batch_size: int, lr: float,
          seed: int) -> dict:
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    try:
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError:  # sentence-transformers < 6
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers

    model = SentenceTransformer(base_model)
    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(out_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=0.1,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=Dataset.from_list(triplets),
        loss=MultipleNegativesRankingLoss(model),
    )
    result = trainer.train()
    model.save(out_dir)
    return {"train_loss": result.training_loss, "train_runtime_s": result.metrics.get("train_runtime"),
            "triplets": len(triplets)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20000, help="HotpotQA train questions")
    p.add_argument("--base-model", default=RAGConfig().embedding_model)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.environ.get("QMR_RUNS_DIR", "runs"), "retriever_minilm"))
    args = p.parse_args(argv)

    from datasets import load_dataset

    rows = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train").shuffle(seed=args.seed)
    triplets = build_triplets(rows.select(range(min(args.n, len(rows)))), seed=args.seed)
    print(f"{len(triplets)} triplets from {args.n} questions")
    metrics = train(triplets, args.base_model, args.out, args.epochs, args.batch_size, args.lr, args.seed)
    with open(os.path.join(args.out, "train_metrics.json"), "w") as f:
        json.dump({**metrics, "args": vars(args)}, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
