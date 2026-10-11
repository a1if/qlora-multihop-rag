"""
src/hotpotqa.py
HotpotQA (distractor setting) loader.

Samples N dev questions with a fixed seed and pools every context paragraph
(2 gold + 8 distractors per question) into one shared retrieval corpus, so
the retriever must find the right paragraphs among ~10 x N candidates.
"""

from dataclasses import dataclass


@dataclass
class HotpotExample:
    id: str
    question: str
    answer: str
    type: str                  # "bridge" | "comparison"
    level: str                 # "easy" | "medium" | "hard"
    gold_titles: list[str]


def load_hotpotqa(n: int, seed: int = 0, split: str = "validation"):
    """Return (examples, corpus) where corpus is a list of {id, title, text} docs."""
    from datasets import load_dataset

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    dataset = dataset.shuffle(seed=seed).select(range(min(n, len(dataset))))

    examples, corpus = [], {}
    for row in dataset:
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"], strict=True):
            corpus.setdefault(title, "".join(sentences).strip())
        examples.append(HotpotExample(
            id=row["id"],
            question=row["question"],
            answer=row["answer"],
            type=row["type"],
            level=row["level"],
            gold_titles=sorted(set(row["supporting_facts"]["title"])),
        ))

    docs = [{"id": i, "title": t, "text": text} for i, (t, text) in enumerate(corpus.items())]
    return examples, docs
