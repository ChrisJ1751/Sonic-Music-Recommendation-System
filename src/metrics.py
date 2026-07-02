"""metrics.py — beyond-accuracy / catalogue-level reporting metrics.

These complement the frozen ranking metrics in `eval_core` (which define what a
"win" is). They are *reporting* metrics: coverage, novelty, diversity, and
distributional inequality. A recommender can be accurate yet narrow — these
quantify that, following Kaminskas & Bridge (TiiS 2017), "Diversity, Serendipity,
Novelty, and Coverage".

`recs` throughout is an (n_users, k) array of recommended item indices.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _valid(recs: np.ndarray) -> np.ndarray:
    """Flatten recs and drop implicit's -1 padding (returned when a list can't fill k)."""
    flat = np.asarray(recs).ravel()
    return flat[flat >= 0]


def catalog_coverage(recs: np.ndarray, n_items: int) -> float:
    """Fraction of the catalogue that appears in at least one user's list."""
    return len(set(_valid(recs).tolist())) / n_items


def novelty(recs: np.ndarray, train: sp.csr_matrix) -> float:
    """Mean self-information of recommended items: -log2(listeners_i / n_users).

    Higher = recommending rarer artists (more novel). Popular artists carry
    little self-information; long-tail artists carry a lot.
    """
    n_users = train.shape[0]
    listeners = np.asarray((train > 0).sum(axis=0)).ravel().astype(float)
    p = np.clip(listeners / n_users, 1e-12, None)
    self_info = -np.log2(p)
    valid = _valid(recs)
    return float(self_info[valid].mean()) if valid.size else 0.0


def gini(recs: np.ndarray, n_items: int) -> float:
    """Gini coefficient of how often each item is recommended (0 = even, 1 = concentrated).

    A high Gini means a few artists soak up most recommendations — the
    popularity-concentration the long tail induces.
    """
    counts = np.bincount(_valid(recs), minlength=n_items).astype(float)
    counts.sort()
    n = counts.size
    if counts.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * counts)) / (n * counts.sum()))


def intra_list_diversity(recs: np.ndarray, item_factors: np.ndarray) -> float:
    """Mean pairwise cosine DISTANCE within each user's list, averaged over users.

    Needs item embeddings (e.g. ALS item factors). 0 = identical items, higher =
    more diverse lists. Model-specific, so the caller supplies the factors.
    """
    f = np.asarray(item_factors, dtype=float)
    norms = np.linalg.norm(f, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = f / norms
    per_user = []
    for row in np.asarray(recs):
        v = unit[row]
        sims = v @ v.T
        k = len(row)
        if k < 2:
            continue
        # mean of off-diagonal cosine distances (1 - sim)
        off = (np.sum(1.0 - sims) - 0.0) / (k * (k - 1))
        per_user.append(off)
    return float(np.mean(per_user)) if per_user else 0.0
