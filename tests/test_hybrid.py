import pytest

from src.retriever import BM25Retriever, HybridRetriever, make_retriever, reciprocal_rank_fusion, tokenize
from tests.conftest import CORPUS

pytest.importorskip("faiss")


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("Who was the 20th President of Rhodes College?") == ["20th", "president", "rhodes", "college"]


def test_bm25_prefers_exact_entity_match():
    bm25 = BM25Retriever(CORPUS)
    hits = bm25.search("Marie Curie", k=3)
    assert hits[0]["title"] == "Marie Curie"
    assert all(h["score"] > 0 for h in hits)


def test_bm25_rare_terms_outweigh_common_ones():
    bm25 = BM25Retriever(CORPUS)
    # "france" appears in three passages, "eiffel" in one
    assert bm25.search("eiffel france", k=1)[0]["title"] == "Eiffel Tower"


def test_bm25_no_match_returns_nothing():
    assert BM25Retriever(CORPUS).search("zzzz qqqq", k=3) == []


def _ids(*ids):
    return [{"id": i, "text": str(i)} for i in ids]


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([_ids(1, 2, 3), _ids(3, 4, 1)], k=4)
    assert [d["id"] for d in fused][:2] == [1, 3]   # in both lists beats top of one
    assert {d["id"] for d in fused} == {1, 2, 3, 4}


def test_rrf_respects_k():
    assert len(reciprocal_rank_fusion([_ids(*range(10)), _ids(*range(5, 15))], k=3)) == 3


def test_hybrid_combines_dense_and_bm25(retriever):
    hybrid = HybridRetriever(retriever, candidates=5)
    hits = hybrid.search("Where was Marie Curie born?", k=2)
    assert hits[0]["title"] == "Marie Curie"
    assert len(hits) == 2 and len({h["id"] for h in hits}) == 2


class FakeReranker:
    """Prefers passages mentioning Warsaw, to show the reranker decides the final order."""

    def predict(self, pairs):
        return [1.0 if "Warsaw" in passage else 0.0 for _, passage in pairs]


def test_reranker_reorders_candidates(retriever):
    hybrid = HybridRetriever(retriever, reranker=FakeReranker(), candidates=5)
    assert hybrid.search("France scientists", k=1)[0]["title"] == "Marie Curie"


def test_make_retriever_kinds(retriever):
    assert make_retriever("dense", retriever) is retriever
    assert isinstance(make_retriever("hybrid", retriever), HybridRetriever)
    with pytest.raises(ValueError):
        make_retriever("sparse-only", retriever)
