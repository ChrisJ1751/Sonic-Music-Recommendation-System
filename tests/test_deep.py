"""Tests for the Mult-VAE deep model (src/deep.py). Skipped if torch is absent."""
import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("torch")
from src import deep  # noqa: E402


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    dense = (rng.random((30, 20)) < 0.3).astype(float)
    for u in range(30):  # every user gets at least two items
        dense[u, u % 20] = 1.0
        dense[u, (u + 5) % 20] = 1.0
    return sp.csr_matrix(dense)


def test_multvae_trains_recommends_and_masks_train():
    train = _toy()
    model = deep.train_multvae(train, epochs=3, hidden=32, latent=8, seed=0)
    recs = deep.multvae_recs(model, train, np.arange(30), k=3)
    assert recs.shape == (30, 3)
    for u in range(30):
        known = set(train.getrow(u).indices.tolist())
        assert known.isdisjoint(set(recs[u].tolist()))


def test_multvae_is_deterministic_given_seed():
    train = _toy()
    a = deep.multvae_recs(deep.train_multvae(train, epochs=3, hidden=16, latent=4, seed=1),
                          train, np.arange(30), k=3)
    b = deep.multvae_recs(deep.train_multvae(train, epochs=3, hidden=16, latent=4, seed=1),
                          train, np.arange(30), k=3)
    assert np.array_equal(a, b)
