"""als_model.py — confidence-weighted implicit ALS (reusable).

Thin, well-documented wrapper around `implicit.als.AlternatingLeastSquares` that
encodes the project's modelling decisions in one place:

  - Implicit feedback (Hu, Koren, Volinsky 2008): split each observed listen
    count into a preference (1 for any listening) and a CONFIDENCE. `implicit`
    reads the matrix values as that confidence, so we build it here (the library
    has no `alpha` argument).
  - **Counts are log-scaled first:** confidence = 1 + alpha * log(1 + count).
    Last.fm listen counts span ~6 orders of magnitude (1 .. 352,698); with raw
    counts a few mega-played artists dominate every confidence and ALS quality
    collapses as alpha rises. The log form is Hu et al.'s own variant for exactly
    this case, and empirically more than doubles NDCG@10 here (see
    notebooks/02 and decisions.md 2026-06-30). `count_transform="linear"` keeps
    raw counts for the ablation.
  - Hyperparameters (factors / regularization / iterations / alpha) come from
    `src/harness/search_config.py`; this module holds no search policy.

Kept separate from the harness so the API (Milestone 6) can reuse it to serve a
single chosen, frozen config.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits

COUNT_TRANSFORMS = ("log1p", "linear")


def to_confidence(train: sp.csr_matrix, alpha: float, count_transform: str = "log1p") -> sp.csr_matrix:
    """Build the confidence matrix c = 1 + alpha * f(count).

    f = log1p (default, for heavy-tailed counts) or identity ("linear"). Same
    sparsity pattern as `train`; only stored values change.
    """
    if count_transform not in COUNT_TRANSFORMS:
        raise ValueError(f"count_transform must be one of {COUNT_TRANSFORMS}; got {count_transform!r}")
    conf = train.copy().astype(np.float32)
    if count_transform == "log1p":
        conf.data = np.log1p(conf.data)
    conf.data = 1.0 + float(alpha) * conf.data
    return conf


def train_als(
    train: sp.csr_matrix,
    factors: int,
    regularization: float,
    iterations: int,
    alpha: float,
    seed: int = 0,
    count_transform: str = "log1p",
) -> tuple[AlternatingLeastSquares, sp.csr_matrix]:
    """Fit ALS on the confidence-weighted train matrix (users x items).

    Returns the fitted model and the confidence matrix used (callers pass that
    same matrix to `recommend_top_n` so already-listened items are filtered).
    BLAS is pinned to one thread for reproducibility and to avoid implicit's
    threadpool performance warning.
    """
    conf = to_confidence(train, alpha, count_transform)
    model = AlternatingLeastSquares(
        factors=int(factors),
        regularization=float(regularization),
        iterations=int(iterations),
        random_state=int(seed),
        use_gpu=False,
    )
    with threadpool_limits(1, "blas"):
        model.fit(conf, show_progress=False)
    return model, conf


def recommend_top_n(
    model: AlternatingLeastSquares,
    user_items: sp.csr_matrix,
    user_rows: np.ndarray,
    n: int = 10,
) -> np.ndarray:
    """Top-n item indices for each user in `user_rows`, best-first.

    Already-listened (train) items are filtered out. `user_items` must be the
    matrix the model was fit on (rows = users). Returns an (len(user_rows), n)
    int array of item-column indices.
    """
    user_rows = np.asarray(user_rows)
    with threadpool_limits(1, "blas"):
        ids, _scores = model.recommend(
            user_rows,
            user_items[user_rows],
            N=n,
            filter_already_liked_items=True,
        )
    return np.asarray(ids)


def recommend_for_user(
    model: AlternatingLeastSquares,
    user_items: sp.csr_matrix,
    user_row: int,
    n: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-n (item indices, scores) for one user, train items filtered. For serving."""
    with threadpool_limits(1, "blas"):
        ids, scores = model.recommend(
            int(user_row),
            user_items[int(user_row)],
            N=n,
            filter_already_liked_items=True,
        )
    return np.asarray(ids), np.asarray(scores)


def similar_items(
    model: AlternatingLeastSquares,
    item_row: int,
    n: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """The n most similar items to `item_row` by ALS item-factor cosine ('fans also like').

    Excludes the query item itself. Returns (item indices, similarity scores).
    """
    with threadpool_limits(1, "blas"):
        ids, scores = model.similar_items(int(item_row), N=n + 1)
    ids, scores = np.asarray(ids), np.asarray(scores)
    keep = ids != int(item_row)
    return ids[keep][:n], scores[keep][:n]
