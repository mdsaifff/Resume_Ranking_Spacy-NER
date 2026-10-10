import pytest

from evaluate import ndcg_at_k, precision_at_k


def test_ndcg_perfect_order_is_one():
    assert ndcg_at_k([3, 2, 1, 0, 0]) == pytest.approx(1.0)


def test_ndcg_penalizes_bad_order():
    assert ndcg_at_k([0, 0, 1, 2, 3]) < ndcg_at_k([3, 0, 0, 2, 1]) < 1.0


def test_ndcg_only_looks_at_top_k():
    assert ndcg_at_k([3, 2, 1, 1, 1, 0], k=5) == pytest.approx(1.0)


def test_ndcg_all_irrelevant_is_zero():
    assert ndcg_at_k([0, 0, 0]) == 0.0


def test_precision_counts_relevance_two_and_up():
    assert precision_at_k([3, 1, 2, 0, 2, 3]) == pytest.approx(3 / 5)
