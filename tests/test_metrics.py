"""Tests for beyond-accuracy reporting metrics (src/metrics.py)."""
import numpy as np
import scipy.sparse as sp

from src import metrics


def test_catalog_coverage():
    recs = np.array([[0, 1, 2], [0, 1, 2]])  # 3 distinct items
    assert metrics.catalog_coverage(recs, n_items=10) == 0.3


def test_novelty_self_information():
    # 2 users; item0 listened by both (p=1 -> 0 bits), item1 by one (p=0.5 -> 1 bit)
    train = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 0.0]]))
    assert abs(metrics.novelty(np.array([[0]]), train) - 0.0) < 1e-9
    assert abs(metrics.novelty(np.array([[1]]), train) - 1.0) < 1e-9


def test_gini_concentration():
    # all recommendations point at one item -> high inequality (n-1)/n
    recs_concentrated = np.array([[0], [0], [0], [0]])
    assert abs(metrics.gini(recs_concentrated, n_items=4) - 0.75) < 1e-9
    # each item recommended equally -> no inequality
    recs_uniform = np.array([[0], [1], [2], [3]])
    assert abs(metrics.gini(recs_uniform, n_items=4) - 0.0) < 1e-9


def test_intra_list_diversity():
    factors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])  # item0,2 identical; item1 orthogonal
    # orthogonal pair -> cosine distance 1.0
    assert abs(metrics.intra_list_diversity(np.array([[0, 1]]), factors) - 1.0) < 1e-9
    # identical pair -> distance 0.0
    assert abs(metrics.intra_list_diversity(np.array([[0, 2]]), factors) - 0.0) < 1e-9
