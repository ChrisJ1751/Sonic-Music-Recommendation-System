"""cold_start.py — fallback for users ALS cannot personalise.

ALS can only recommend for a user it has a learned latent vector for — i.e. a
user with >=1 interaction in the trained matrix. Two populations need a fallback:

  - users entirely absent from training (a brand-new user_id hitting the API), and
  - users with too few interactions to have a meaningful vector.

The fallback is **popularity by distinct listeners**, not by total play count.
Play counts are dominated by a few heavy users (counts to 352k), so ranking by
raw plays would surface whatever a handful of superfans hammered; ranking by how
MANY distinct users listened is the robust "what's broadly liked" signal, and is
the honest reference the personalised model must beat (see notebooks/02).

The EDA showed the user-side cold population here is tiny (~8 users with <2
interactions), so this is mostly the API's zero-interaction safety net rather
than a large segment. A tag/content variant is possible future work; popularity
is the dependable default.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def item_popularity(train: sp.csr_matrix) -> np.ndarray:
    """Distinct-listener count per item (column), the robust popularity signal."""
    return np.asarray((train > 0).sum(axis=0)).ravel()


def popular_items(train: sp.csr_matrix, k: int = 10, exclude: set[int] | None = None) -> np.ndarray:
    """Top-k item indices by distinct listeners, best-first, skipping `exclude`.

    `exclude` is the set of item columns the user already has (so we don't
    recommend something they've heard). Stops as soon as k items are collected.
    """
    order = np.argsort(item_popularity(train))[::-1].tolist()
    if not exclude:
        return np.array(order[:k])
    out: list[int] = []
    for i in order:
        if i not in exclude:
            out.append(i)
            if len(out) == k:
                break
    return np.array(out)


def recommend_cold_start(train: sp.csr_matrix, k: int = 10, exclude: set[int] | None = None) -> np.ndarray:
    """Default cold-start recommendation: global popularity (distinct listeners)."""
    return popular_items(train, k=k, exclude=exclude)
