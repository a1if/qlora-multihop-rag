import re

import pytest

from configs.training_config import RAGConfig
from src.data_sft import build_examples, order_gold, read_jsonl, stats, write_jsonl
from src.rag_pipeline import RAGPipeline
from src.text_utils import ABSTAIN, extract_answer, extract_query, parse_sources

TITLES = ["Glenn Quagmire", "Seth MacFarlane", "Family Guy", "Logan Lucky", "Daron Sutton",
          "Tiegs for Two", "Anthony Frewin", "Ted (film)", "Peter Griffin", "Fox Broadcasting"]


def make_row(i=0, question="Who directed the 2012 film voiced by Glenn Quagmire's voice actor?", answer="Seth MacFarlane"):
    return {
        "id": f"q{i}",
        "question": question,
        "answer": answer,
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": ["Glenn Quagmire", "Seth MacFarlane", "Glenn Quagmire"], "sent_id": [0, 1, 2]},
        "context": {"title": TITLES, "sentences": [[f"{t} sentence one.", f" {t} sentence two."] for t in TITLES]},
    }


ROWS = [make_row(i) for i in range(200)]
GOLD = {"Glenn Quagmire", "Seth MacFarlane"}


def passage_titles(instruction: str) -> list[str]:
    return re.findall(r"^\[\d+\] (.+?): ", instruction, flags=re.M)


@pytest.fixture(scope="module")
def examples():
    return build_examples(ROWS, k=5, max_chars=700, abstain_ratio=0.2, query_ratio=0.5, seed=0)


def test_task_mix(examples):
    counts = stats(examples)["tasks"]
    assert counts["reader"] + counts["abstain"] == len(ROWS)
    assert 20 <= counts["abstain"] <= 60          # ~20% of 200
    assert 70 <= counts["query"] <= 130           # ~50% of 200


def test_reader_has_both_gold_and_cites_them(examples):
    for ex in (e for e in examples if e.task == "reader"):
        titles = passage_titles(ex.instruction)
        assert len(titles) == 5 and GOLD <= set(titles)
        cited = {titles[i - 1] for i in parse_sources(ex.response)}
        assert cited == GOLD
        assert extract_answer(ex.response) == "Seth MacFarlane"


def test_gold_positions_are_shuffled(examples):
    positions = {tuple(parse_sources(e.response)) for e in examples if e.task == "reader"}
    assert len(positions) > 3


def test_abstain_examples_lack_full_support(examples):
    for ex in (e for e in examples if e.task == "abstain"):
        titles = passage_titles(ex.instruction)
        assert len(titles) == 5 and not GOLD <= set(titles)
        assert extract_answer(ex.response) == ABSTAIN
        assert parse_sources(ex.response) == []


def test_query_target_is_bridge_entity(examples):
    for ex in (e for e in examples if e.task == "query"):
        titles = passage_titles(ex.instruction)
        assert len(titles) == 3 and "Glenn Quagmire" in titles and "Seth MacFarlane" not in titles
        assert extract_query(ex.response) == "Seth MacFarlane"


def test_order_gold_prefers_title_named_in_question():
    assert order_gold("Who voices Glenn Quagmire?", ["Seth MacFarlane", "Glenn Quagmire"]) == (
        "Glenn Quagmire", "Seth MacFarlane")
    # No title in the question: fall back to word overlap.
    assert order_gold("Which band did the singer of Wonderwall join?", ["Oasis (band)", "Wonderwall (song)"])[0] == \
        "Wonderwall (song)"


def test_prompts_match_inference(examples, fake_llm, retriever):
    """Training prompts must be byte-identical to what the pipeline sends."""
    pipeline = RAGPipeline(fake_llm, retriever, RAGConfig())
    reader = next(e for e in examples if e.task == "reader")
    question = ROWS[0]["question"]
    titles = passage_titles(reader.instruction)
    passages = [{"title": t, "text": f"{t} sentence one. {t} sentence two."} for t in titles]
    pipeline._read(question, passages, short=True)
    assert fake_llm.calls[-1] == reader.instruction


def test_deterministic_and_round_trips(examples, tmp_path):
    again = build_examples(ROWS, k=5, max_chars=700, abstain_ratio=0.2, query_ratio=0.5, seed=0)
    assert again == examples
    write_jsonl(tmp_path / "train.jsonl", examples)
    assert read_jsonl(tmp_path / "train.jsonl") == examples


def test_malformed_rows_skipped():
    bad = make_row()
    bad["supporting_facts"] = {"title": ["Glenn Quagmire"], "sent_id": [0]}
    assert build_examples([bad], k=5, max_chars=700, abstain_ratio=0, query_ratio=1) == []


def test_retriever_triplets_use_gold_and_hard_negatives():
    from src.finetune_retriever import build_triplets

    triplets = build_triplets(ROWS[:50], seed=0)
    assert len(triplets) == 100                       # one per gold passage
    for t in triplets:
        assert t["anchor"] == ROWS[0]["question"]
        assert t["positive"].split(":")[0] in GOLD    # "title: text", like the index
        assert t["negative"].split(":")[0] not in GOLD
