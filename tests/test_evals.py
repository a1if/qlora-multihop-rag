"""
Regression eval over the 25-question golden set (evals/).

- BM25 retrieval runs live (no model needed, so it runs in CI).
- Answer quality, faithfulness and citations are checked on recorded outputs
  of the real pipeline (evals/recorded/*.jsonl, produced by evals/record.py).
- With QMR_LIVE_EVAL=1 and a GPU, test_live re-runs the real model via the MCP server.
"""

import os

import pytest

from evals.checks import HERE, answer_supported, corpus, evaluate, golden, load_jsonl, thresholds
from src.retriever import BM25Retriever

GATES = thresholds()
RECORDED = sorted(f for f in os.listdir(os.path.join(HERE, "recorded")) if f.endswith(".jsonl")) \
    if os.path.isdir(os.path.join(HERE, "recorded")) else []


def test_golden_set_is_consistent():
    gold, titles = golden(), {p["title"] for p in corpus()}
    assert 20 <= len(gold) <= 30
    assert len({g["id"] for g in gold}) == len(gold)
    for g in gold:
        assert len(g["gold_titles"]) == 2 and set(g["gold_titles"]) <= titles


@pytest.mark.parametrize("k,gate_all,gate_any", [(5, "k5_all_gold", "k5_any_gold"), (10, "k10_all_gold", None)])
def test_bm25_retrieval_hit_rate(k, gate_all, gate_any):
    bm25 = BM25Retriever(corpus())
    all_hit = any_hit = 0
    for g in golden():
        titles = {h["title"] for h in bm25.search(g["question"], k)}
        all_hit += set(g["gold_titles"]) <= titles
        any_hit += bool(set(g["gold_titles"]) & titles)
    n = len(golden())
    assert all_hit / n >= GATES["bm25"][gate_all], f"all-gold@{k} = {all_hit}/{n}"
    if gate_any:
        assert any_hit / n >= GATES["bm25"][gate_any], f"any-gold@{k} = {any_hit}/{n}"


def test_faithfulness_check_itself():
    passages = [{"title": "Seth MacFarlane", "text": "He directed the 2012 film Ted."}]
    assert answer_supported("Ted", passages)
    assert not answer_supported("Family Guy", passages)
    assert not answer_supported("Ted", [])
    assert answer_supported("yes", passages * 2) and not answer_supported("no", passages)
    # possessives in the evidence still support the bare name
    assert answer_supported("Stephen King", [{"title": "The Shining", "text": "based on Stephen King's novel"}])


@pytest.mark.parametrize("name", RECORDED or [pytest.param(None, marks=pytest.mark.skip("no recorded outputs yet"))])
def test_recorded_outputs_meet_gates(name):
    stem = name[:-len(".jsonl")]
    gates = GATES["recorded"].get(stem)
    assert gates, f"add gates for {stem} to evals/thresholds.json"
    metrics = evaluate(load_jsonl(os.path.join(HERE, "recorded", name)), golden())
    failures = []
    for gate, limit in gates.items():          # gates look like "min_em": 0.5 or "max_wrong_when_evidence_missing": 0.4
        if gate.startswith("_"):                # notes such as "_measured"
            continue
        kind, metric = gate.split("_", 1)
        value = metrics[metric]
        if (kind == "min" and value < limit) or (kind == "max" and value > limit):
            failures.append(f"{metric}={value:.3f} ({kind} {limit})")
    assert not failures, f"{stem}: " + ", ".join(failures)


@pytest.mark.skipif(os.environ.get("QMR_LIVE_EVAL") != "1", reason="set QMR_LIVE_EVAL=1 to run the real model")
def test_live():
    import anyio

    from configs.training_config import RAGConfig
    from evals.record import record
    from src.mcp_server import create_server, lazy_pipeline
    from src.rag_pipeline import load_retriever

    config = RAGConfig(llm=os.environ["QMR_LLM"], retriever=os.environ.get("QMR_RETRIEVER", "hybrid_rerank"),
                       index_dir=os.environ["QMR_INDEX_DIR"])
    retriever = load_retriever(config)
    server = create_server(retriever, config, lazy_pipeline(retriever, config))
    metrics = evaluate(anyio.run(record, server, golden()), golden())
    assert metrics["faithfulness"] >= 0.85 and metrics["em"] >= 0.4, metrics
