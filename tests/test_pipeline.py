import threading

import pytest

from src.rag_pipeline import RAGPipeline
from tests.conftest import QUESTION

pytest.importorskip("faiss")


def titles(result):
    return [p["title"] for p in result.passages]


def test_closed_book_uses_no_retrieval(pipeline):
    result = pipeline.answer(QUESTION, mode="closed")
    assert result.passages == [] and result.queries == []
    assert result.answer == "France"
    assert result.timings["llm_calls"] == 1
    assert result.timings["retrieval_s"] == 0.0


def test_single_hop_misses_bridge_passage(pipeline):
    result = pipeline.answer(QUESTION, mode="single")
    assert "Marie Curie" not in titles(result)
    assert result.answer == "France"


def test_iterative_finds_bridge_and_answers(pipeline):
    result = pipeline.answer(QUESTION, mode="iterative")
    assert result.queries == [QUESTION, "Marie Curie birthplace"]
    assert "Marie Curie" in titles(result)
    assert result.answer == "Poland"          # <think> block stripped
    assert result.timings["llm_calls"] == 2   # follow-up + reader


def test_decompose_retrieves_per_subquestion(pipeline):
    result = pipeline.answer(QUESTION, mode="decompose")
    assert result.queries == ["Who discovered polonium?", "Where was Marie Curie born?"]
    assert "Marie Curie" in titles(result)
    assert result.answer == "Poland"


@pytest.mark.parametrize("mode", ["single", "iterative", "decompose"])
def test_every_mode_respects_evidence_budget(pipeline, mode):
    assert len(pipeline.answer(QUESTION, mode=mode, k=3).passages) <= 3


def test_long_answer_keeps_full_text(pipeline):
    result = pipeline.answer(QUESTION, mode="iterative", short=False)
    assert result.answer == "Answer: Poland"


def test_reader_prompt_contains_only_generated_text(pipeline, fake_llm):
    pipeline.answer(QUESTION, mode="single")
    reader_prompt = fake_llm.calls[-1]
    assert reader_prompt.count("Question:") == 1
    assert "[1] " in reader_prompt


def test_unknown_mode_rejected(pipeline):
    with pytest.raises(ValueError):
        pipeline.answer(QUESTION, mode="bogus")


def test_retrieval_mode_without_retriever(fake_llm, config):
    with pytest.raises(ValueError):
        RAGPipeline(fake_llm, None, config).answer(QUESTION, mode="single")


def test_timings_are_isolated_per_thread(pipeline):
    results = {}

    def run(mode):
        results[mode] = pipeline.answer(QUESTION, mode=mode)

    threads = [threading.Thread(target=run, args=(m,)) for m in ("closed", "iterative", "decompose")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results["closed"].timings["llm_calls"] == 1
    assert results["iterative"].timings["llm_calls"] == 2
    assert results["decompose"].timings["llm_calls"] == 2
