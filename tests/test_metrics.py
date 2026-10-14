import pytest

from src.metrics import (
    all_support_found, bootstrap_ci, exact_match, f1_score, normalize_answer, paired_bootstrap_delta,
    support_recall,
)


def test_normalize_answer():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"


@pytest.mark.parametrize("pred,gold,em", [
    ("the Eiffel Tower", "Eiffel Tower", 1.0),
    ("Eiffel", "Eiffel Tower", 0.0),
    ("YES", "yes", 1.0),
])
def test_exact_match(pred, gold, em):
    assert exact_match(pred, gold) == em


def test_f1_partial_overlap():
    # pred tokens {marie, curie}, gold {marie, sklodowska, curie}: P=1, R=2/3
    assert f1_score("Marie Curie", "Marie Sklodowska Curie") == pytest.approx(0.8)


def test_f1_yes_no_must_match_exactly():
    assert f1_score("no", "yes") == 0.0
    assert f1_score("yes it is", "yes") == 0.0


def test_f1_no_overlap():
    assert f1_score("Paris", "Warsaw") == 0.0


def test_support_metrics():
    assert support_recall(["A", "C"], ["A", "B"]) == 0.5
    assert all_support_found(["A", "B", "C"], ["A", "B"]) == 1.0
    assert all_support_found(["A"], ["A", "B"]) == 0.0
    assert support_recall(["A"], []) == 0.0


def test_bootstrap_ci_brackets_mean():
    mean, lo, hi = bootstrap_ci([0, 1] * 50)
    assert mean == 0.5
    assert lo < 0.5 < hi


def test_paired_delta_detects_consistent_gain():
    a, b = [1.0] * 40 + [0.0] * 10, [0.0] * 50
    delta, lo, hi = paired_bootstrap_delta(a, b)
    assert delta == pytest.approx(0.8)
    assert lo > 0


def test_paired_delta_requires_same_questions():
    with pytest.raises(ValueError):
        paired_bootstrap_delta([1, 0], [1])
