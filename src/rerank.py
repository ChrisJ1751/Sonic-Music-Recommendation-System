"""rerank.py — Maximal Marginal Relevance (MMR) diversity re-ranking.

Carbonell & Goldstein (SIGIR 1998). The model card and notebook 04 show the
served model is strongly popularity-biased; MMR is the standard lever to trade a
little accuracy for a more varied list. Given a pool of candidates with
relevance scores and item embeddings, MMR greedily builds a list that balances
relevance against dissimilarity to what's already chosen:

    pick argmax_i  lambda * rel(i)  -  (1 - lambda) * max_{j in selected} sim(i, j)

`lambda_ = 1` is pure relevance (original ranking); lower `lambda_` favours
diversity. The API exposes this as `diversity = 1 - lambda_`.
"""
from __future__ import annotations

import numpy as np


def mmr_rerank(items: np.ndarray, scores: np.ndarray, factors: np.ndarray,
               k: int, lambda_: float = 0.7) -> np.ndarray:
    """Re-rank candidate `items` (with `scores` and row-aligned `factors`) by MMR.

    Returns the top-k item ids in MMR order. `factors` is (n_candidates, dim).
    """
    items = np.asarray(items)
    scores = np.asarray(scores, dtype=float)
    if len(items) == 0:
        return items
    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError("lambda_ must be in [0, 1]")

    # relevance min-maxed to [0, 1] so it's comparable to cosine similarity
    rng = scores.max() - scores.min()
    rel = (scores - scores.min()) / rng if rng > 0 else np.zeros_like(scores)

    f = np.asarray(factors, dtype=float)
    norms = np.linalg.norm(f, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = f / norms
    sim = unit @ unit.T  # candidate-candidate cosine similarity

    k = min(k, len(items))
    selected: list[int] = []
    remaining = set(range(len(items)))
    for _ in range(k):
        best_idx, best_val = None, -np.inf
        for i in remaining:
            penalty = max((sim[i, j] for j in selected), default=0.0)
            val = lambda_ * rel[i] - (1.0 - lambda_) * penalty
            if val > best_val:
                best_val, best_idx = val, i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return items[selected]
