"""End-to-end run of the benchmark harness on a toy dataset with fake models."""

import json

import pytest

from src import eval_hotpotqa
from src.hotpotqa import HotpotExample
from tests.conftest import CORPUS, QUESTION, FakeLLM

pytest.importorskip("faiss")
pytest.importorskip("tqdm")

EXAMPLES = [
    HotpotExample("q1", QUESTION, "Poland", "bridge", "hard", ["Marie Curie", "Polonium"]),
    HotpotExample("q2", "Which country is the Eiffel Tower in?", "France", "comparison", "easy", ["Eiffel Tower"]),
]


@pytest.fixture
def fake_backend(monkeypatch, embedder):
    from src.retriever import DenseRetriever

    llms = []

    def make_fake_llm(spec, *args, **kwargs):
        llms.append(FakeLLM())
        return llms[-1]

    monkeypatch.setattr("src.llm.make_llm", make_fake_llm)
    monkeypatch.setattr(eval_hotpotqa, "build_retriever", lambda *a, **k: DenseRetriever.build(CORPUS, embedder))
    return llms


def run(tmp_path, *extra):
    argv = ["--n", "2", "--top-k", "2", "--out", str(tmp_path), "--reference", "base:single",
            "--model", "base=openai:fake@http://llm.test/v1",
            "--systems", "base:closed", "base:single", "base:iterative", *extra]
    return eval_hotpotqa.main(argv, examples=EXAMPLES, docs=CORPUS)


def test_harness_scores_and_writes_reports(tmp_path, fake_backend):
    rows = {r["system"]: r for r in run(tmp_path, "--workers", "2")}

    # q1 needs the bridge passage; q2's gold answer happens to be the fallback "France".
    assert rows["base:closed"]["EM"] == 0.5
    assert rows["base:single"]["EM"] == 0.5
    assert rows["base:iterative"]["EM"] == 1.0
    assert rows["base:iterative"]["EM_bridge"] == 1.0
    assert rows["base:iterative"]["dF1_vs_ref"] == pytest.approx(0.5)
    assert rows["base:iterative"]["llm_calls"] == 2.0
    assert rows["base:iterative"]["throughput_qps"] > 0 and rows["base:iterative"]["workers"] == 2
    assert "SupportRecall@k" not in rows["base:closed"]

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "`base:iterative`" in summary
    assert (tmp_path / "summary.csv").exists()
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["models"]["base"] == "openai:fake@http://llm.test/v1"


def test_harness_resumes_without_recomputing(tmp_path, fake_backend):
    run(tmp_path)
    first_calls = len(fake_backend[0].calls)
    assert first_calls > 0

    rows = run(tmp_path)
    assert len(fake_backend) == 1, "no model should be loaded when every prediction exists"
    assert {r["system"] for r in rows} == {"base:closed", "base:single", "base:iterative"}


def test_unknown_model_in_system_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        eval_hotpotqa.parse_args(["--systems", "gpt:single"])
