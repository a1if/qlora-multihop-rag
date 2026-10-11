import pytest

from src.retriever import DenseRetriever, merge_round_robin
from tests.conftest import CORPUS

pytest.importorskip("faiss")


def _docs(*ids):
    return [{"id": i, "text": str(i)} for i in ids]


def test_merge_round_robin_interleaves_and_dedupes():
    merged = merge_round_robin([_docs(1, 2, 3), _docs(2, 4, 5)], k=4)
    assert [d["id"] for d in merged] == [1, 2, 4, 3]


def test_merge_round_robin_short_lists():
    assert [d["id"] for d in merge_round_robin([_docs(1), _docs()], k=5)] == [1]


def test_search_ranks_relevant_doc_first(retriever):
    hits = retriever.search("Where was Marie Curie born?", k=2)
    assert hits[0]["title"] == "Marie Curie"
    assert hits[0]["score"] >= hits[1]["score"]


def test_scores_are_cosine(retriever):
    # Searching with a document's own text should give similarity ~1.
    from src.retriever import doc_text

    hit = retriever.search(doc_text(CORPUS[4]), k=1)[0]
    assert hit["id"] == 4
    assert hit["score"] == pytest.approx(1.0, abs=1e-5)


def test_save_and_load_round_trip(retriever, embedder, tmp_path):
    retriever.save(tmp_path)
    assert DenseRetriever.exists(tmp_path)
    loaded = DenseRetriever.load(tmp_path, embedder)
    assert loaded.docs == CORPUS
    assert loaded.search("polonium", 1)[0]["title"] == "Polonium"
