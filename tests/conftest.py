"""Shared test fixtures.

The important one is `synthetic_state`: a genuine `serving.RecoState` built from
a tiny seeded matrix through the *real* `models.fit_ease` and
`als_model.train_als`. That lets the API contract tests exercise real serving
code without the 514 MiB `ease_B.npy` artifact, so they run in CI on every push.
`tests/test_api.py` still covers the real 360K core and is marked `integration`.

Sizing here is deliberate, not arbitrary. `serving.recommend` builds an MMR
candidate pool of `max(k * 6, 60)` items, and `rerank.mmr_rerank` min-maxes the
candidate scores -- so if that pool ever reaches n_items it sweeps in the -inf
already-played mask and every MMR weight becomes NaN. 120 items keeps the pool
strictly below the catalogue for the k values these tests use.

`reg` is 1.0 rather than serving's 100.0 for the same reason: at 80x120 the Gram
matrix is small enough that reg=100 would swamp it and collapse B to ~0, leaving
degenerate all-equal scores that no ordering assertion could detect.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient

from src import als_model, models, serving
from src.data_loading import InteractionMatrix

N_USERS = 80
N_ITEMS = 120
SEED = 0
TOY_EASE_REG = 1.0


def _toy_interaction_matrix() -> sp.csr_matrix:
    """80 users x 120 artists, 6-10 artists each, popularity-skewed like real data."""
    rng = np.random.default_rng(SEED)
    weights = 1.0 / np.arange(1, N_ITEMS + 1)   # Zipf-ish: item 0 most listened
    weights /= weights.sum()

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for u in range(N_USERS):
        n = int(rng.integers(6, 11))
        items = rng.choice(N_ITEMS, size=n, replace=False, p=weights)
        for c in items:
            rows.append(u)
            cols.append(int(c))
            vals.append(float(rng.integers(1, 500)))   # play counts
    return sp.csr_matrix((vals, (rows, cols)), shape=(N_USERS, N_ITEMS), dtype=np.float32)


@pytest.fixture(scope="session")
def synthetic_state() -> serving.RecoState:
    """A real RecoState over toy data -- same code path as production, no artifacts."""
    matrix = _toy_interaction_matrix()
    item_ids = np.array([f"artist-{i:03d}" for i in range(N_ITEMS)])
    user_ids = np.arange(N_USERS)

    im = InteractionMatrix(
        matrix=matrix,
        user_ids=user_ids,
        item_ids=item_ids,
        user_pos={int(u): i for i, u in enumerate(user_ids)},
        item_pos={str(a): i for i, a in enumerate(item_ids)},
    )
    B = models.fit_ease(matrix, reg=TOY_EASE_REG)
    als, _ = als_model.train_als(matrix, factors=16, regularization=0.01,
                                 iterations=5, alpha=1.0, seed=SEED)
    Xbin = (matrix > 0).astype(np.float32).tocsr()
    return serving.RecoState(dataset="synthetic", im=im, B=B, Xbin=Xbin, als=als)


@pytest.fixture()
def api_client(synthetic_state):
    """TestClient with state injected, bypassing the artifact-loading lifespan.

    Deliberately NOT used as a context manager: entering TestClient runs the
    lifespan, which calls serving.load_state() and would need the real 360K core
    plus the 514 MiB EASE matrix.
    """
    from api import main

    sentinel = object()
    previous = main.STATE.get("reco", sentinel)
    main.STATE["reco"] = synthetic_state
    client = TestClient(main.app)
    try:
        yield client
    finally:
        if previous is sentinel:
            main.STATE.pop("reco", None)
        else:
            main.STATE["reco"] = previous
