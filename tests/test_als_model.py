"""Tests for the ALS wrapper (src/als_model.py).

These check the confidence transform math and that the implicit wrapper fits and
recommends sanely (shape, train-item filtering, determinism). They do not assert
metric *values* on real data — that's the search's job, logged, not pinned.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from src import als_model


def test_to_confidence_log1p_and_linear():
    m = sp.csr_matrix(np.array([[9.0, 0.0], [0.0, 1.0]]))
    # log1p: c = 1 + alpha*log(1+count); count=9 -> log(10)=2.302585
    c = als_model.to_confidence(m, alpha=2.0, count_transform="log1p")
    assert abs(c[0, 0] - (1.0 + 2.0 * np.log1p(9.0))) < 1e-5
    # linear: c = 1 + alpha*count
    c2 = als_model.to_confidence(m, alpha=2.0, count_transform="linear")
    assert abs(c2[0, 0] - (1.0 + 2.0 * 9.0)) < 1e-5
    # sparsity pattern preserved
    assert c.nnz == m.nnz


def test_to_confidence_rejects_unknown_transform():
    m = sp.csr_matrix(np.array([[1.0]]))
    with pytest.raises(ValueError):
        als_model.to_confidence(m, alpha=1.0, count_transform="sqrt")


def test_train_and_recommend_shapes_and_filtering():
    # 6 users, 8 items, block structure so there is something to learn
    rng = np.random.default_rng(0)
    dense = (rng.random((6, 8)) < 0.5).astype(float) * rng.integers(1, 50, (6, 8))
    # guarantee every user has >=2 items
    for u in range(6):
        if dense[u].sum() == 0:
            dense[u, u % 8] = 5.0
            dense[u, (u + 1) % 8] = 3.0
    train = sp.csr_matrix(dense)

    model, conf = als_model.train_als(train, factors=4, regularization=0.05,
                                      iterations=5, alpha=1.0, seed=0)
    users = np.arange(6)
    recs = als_model.recommend_top_n(model, conf, users, n=3)
    assert recs.shape == (6, 3)
    # recommendations must exclude items the user already has in train
    for u in users:
        train_items = set(train.getrow(u).indices.tolist())
        assert train_items.isdisjoint(set(recs[u].tolist()))


def test_recommend_is_deterministic_given_seed():
    rng = np.random.default_rng(1)
    train = sp.csr_matrix((rng.random((8, 10)) < 0.5).astype(float) * 5)
    a, ca = als_model.train_als(train, 4, 0.05, 5, 1.0, seed=7)
    b, cb = als_model.train_als(train, 4, 0.05, 5, 1.0, seed=7)
    ra = als_model.recommend_top_n(a, ca, np.arange(8), n=3)
    rb = als_model.recommend_top_n(b, cb, np.arange(8), n=3)
    assert np.array_equal(ra, rb)
