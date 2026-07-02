"""Smoke tests for the model zoo + comparison driver (src/models.py).

These check shapes, train-item filtering, and that the comparison runs and ranks;
they don't pin metric values (that's the search/comparison's job, logged).
"""
import numpy as np
import pytest
import scipy.sparse as sp

from src import models


def _toy_train(seed=0):
    rng = np.random.default_rng(seed)
    dense = (rng.random((12, 15)) < 0.4).astype(float) * rng.integers(1, 50, (12, 15))
    for u in range(12):  # ensure every user has >=2 items
        dense[u, u % 15] = 5.0
        dense[u, (u + 3) % 15] = 3.0
    return sp.csr_matrix(dense)


def test_popularity_excludes_train_and_shape():
    train = _toy_train()
    rows = np.arange(12)
    recs = models.popularity_recs(train, rows, k=3)
    assert recs.shape == (12, 3)
    for u in rows:
        known = set(train.getrow(int(u)).indices.tolist())
        assert known.isdisjoint(set(recs[u].tolist()))


@pytest.mark.parametrize("fn", [models.als_recs, models.bpr_recs, models.itemitem_bm25_recs])
def test_each_model_runs_and_shape(fn):
    train = _toy_train()
    recs = fn(train, np.arange(12), k=3, seed=0)
    assert recs.shape == (12, 3)


def test_ease_runs_and_excludes_train():
    train = _toy_train()
    recs = models.ease_recs(train, np.arange(12), k=3, reg=10.0)
    assert recs.shape == (12, 3)
    for u in range(12):
        known = set(train.getrow(u).indices.tolist())
        assert known.isdisjoint(set(recs[u].tolist()))


def test_compare_runs_and_returns_all_models():
    train = _toy_train(0)
    test = _toy_train(1)  # disjoint-ish small "test" just to exercise the driver
    rows = models.compare(train, test, k=3, seeds=(0,))
    assert {r["model"] for r in rows} == set(models.MODELS)
    assert all("ndcg@3" in r and "novelty" in r and "gini" in r for r in rows)
