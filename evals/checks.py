"""
evals/checks.py
Deterministic quality checks for recorded or live pipeline outputs.

A record is what the MCP `answer` tool returns for one golden question:
    {id, answer, queries, passages: [{id, title, text}], citations: [passage ids]}

Checks:
    retrieval hit rate   both gold paragraphs among the evidence (and at least one)
    citation precision   cited passages that are gold supporting paragraphs
    faithfulness         every content word of the answer appears in a cited passage
                         (yes/no answers: at least two passages cited, as comparisons need both)
    abstention           "insufficient evidence" should only happen when gold is missing
    EM / F1              official HotpotQA answer normalisation
"""

import json
import os
import re

from src.metrics import exact_match, f1_score, normalize_answer
from src.text_utils import is_abstention

HERE = os.path.dirname(os.path.abspath(__file__))


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def golden() -> list[dict]:
    return load_jsonl(os.path.join(HERE, "golden.jsonl"))


def corpus() -> list[dict]:
    return load_jsonl(os.path.join(HERE, "corpus.jsonl"))


def thresholds() -> dict:
    with open(os.path.join(HERE, "thresholds.json")) as f:
        return json.load(f)


def _words(text: str) -> set[str]:
    # drop possessives first, or "Stephen King's novel" normalises to "kings" and misses "King"
    text = re.sub(r"['’]s\b", "", text)
    return set(re.findall(r"[a-z0-9]+", normalize_answer(text)))


def answer_supported(answer: str, cited: list[dict]) -> bool:
    """True if the answer is grounded in the passages it cites."""
    norm = normalize_answer(answer)
    if norm in ("yes", "no"):
        return len(cited) >= 2
    if not cited:
        return False
    evidence = set().union(*(_words(f"{p.get('title', '')} {p['text']}") for p in cited))
    words = _words(answer)
    return bool(words) and words <= evidence


def evaluate(records: list[dict], gold: list[dict]) -> dict:
    by_id = {g["id"]: g for g in gold}
    n = len(records)
    em = f1 = all_hit = any_hit = 0.0
    answered = faithful = abstained = abstained_ok = missing = wrong_when_missing = 0
    cite_precision, cite_n = 0.0, 0
    for r in records:
        g = by_id[r["id"]]
        titles = {p["title"] for p in r["passages"]}
        gold_titles = set(g["gold_titles"])
        full = gold_titles <= titles
        all_hit += full
        any_hit += bool(gold_titles & titles)
        em += exact_match(r["answer"], g["answer"])
        f1 += f1_score(r["answer"], g["answer"])
        missing += not full

        if is_abstention(r["answer"]):
            abstained += 1
            abstained_ok += not full
            continue
        answered += 1
        if not full and not exact_match(r["answer"], g["answer"]):
            wrong_when_missing += 1
        by_pid = {p["id"]: p for p in r["passages"]}
        cited = [by_pid[c] for c in r.get("citations", []) if c in by_pid]
        faithful += answer_supported(r["answer"], cited)
        if cited:
            cite_precision += sum(p["title"] in gold_titles for p in cited) / len(cited)
            cite_n += 1

    return {
        "questions": n,
        "em": em / n,
        "f1": f1 / n,
        "retrieval_all_gold": all_hit / n,
        "retrieval_any_gold": any_hit / n,
        "faithfulness": faithful / answered if answered else 1.0,
        "citation_precision": cite_precision / cite_n if cite_n else 0.0,
        "abstention_rate": abstained / n,
        "abstentions_justified": abstained_ok / abstained if abstained else 1.0,
        "wrong_when_evidence_missing": wrong_when_missing / missing if missing else 0.0,
    }
