"""
src/metrics.py
Answer and retrieval metrics for HotpotQA-style evaluation.

Answer EM / F1 follow the official HotpotQA evaluation script
(https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).
"""

import re
import string
from collections import Counter

import numpy as np

_PUNCT = set(string.punctuation)


# ─── Answer Metrics ───────────────────────────────────────────────────────────

def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(gold)

    # yes/no answers must match exactly (as in the official script)
    special = {"yes", "no", "noanswer"}
    if (pred in special or ref in special) and pred != ref:
        return 0.0

    pred_tokens, ref_tokens = pred.split(), ref.split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ─── Retrieval Metrics ────────────────────────────────────────────────────────

def support_recall(retrieved_titles: list[str], gold_titles: list[str]) -> float:
    """Fraction of gold supporting paragraphs present in the retrieved set."""
    if not gold_titles:
        return 0.0
    return len(set(gold_titles) & set(retrieved_titles)) / len(set(gold_titles))


def all_support_found(retrieved_titles: list[str], gold_titles: list[str]) -> float:
    """1.0 if every gold supporting paragraph was retrieved, else 0.0."""
    return float(bool(gold_titles) and set(gold_titles) <= set(retrieved_titles))


# ─── Confidence Intervals ─────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Mean and percentile-bootstrap (1 - alpha) confidence interval."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap_delta(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """
    Mean difference (a - b) over the same questions, with a paired bootstrap CI.
    If the interval excludes 0 the difference is significant at level alpha.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_bootstrap_delta needs scores for the same questions")
    return bootstrap_ci(a - b, n_boot=n_boot, alpha=alpha, seed=seed)
