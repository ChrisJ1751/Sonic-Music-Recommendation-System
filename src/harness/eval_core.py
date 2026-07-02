"""eval_core.py — THE FROZEN CORRECTNESS LAYER (one of three).

>>> The search loop NEVER edits this file. <<<

It defines two things and only two things:
  1. The per-user train/test interaction split.
  2. The ranking metrics: precision@k, recall@k, NDCG@k.

These define what a "win" means. If the thing trying to win could edit them, the
search would be unfalsifiable. So this file is frozen: the runner imports it; it
does not mutate it. Editing it is a model-integrity bug, not a tuning choice.

These functions are validated against a hand-built toy example in
`tests/test_eval_core.py` — a tiny case whose correct precision@k / recall@k /
NDCG@k were computed by hand — so we trust them before they touch real data.

Why a per-user holdout split (not a naive row split): see
`docs/specs/eval_design.md`. The one-line version: to recommend *for* a user the
model must learn that user's latent vector from their TRAIN interactions; a
random row split leaks held-out artists into that vector and inflates the score,
while splitting whole users out makes ALS unable to score them at all.

Signatures are frozen (committed Milestone 1). Bodies implemented and toy-tested
in Milestone 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class UserHoldoutSplit:
    """Result of a per-user interaction split.

    train: CSR (n_users x n_items) — interactions the model learns from.
    test:  CSR (n_users x n_items) — held-out interactions, same user rows.
    Every test interaction's user has >=1 remaining train interaction (so the
    model can form a user vector), and the two matrices are disjoint.
    """
    train: sp.csr_matrix
    test: sp.csr_matrix
    seed: int
    test_fraction: float
    min_train_interactions: int


def per_user_train_test_split(
    interactions: sp.csr_matrix,
    test_fraction: float = 0.2,
    min_train_interactions: int = 1,
    min_user_interactions: int = 2,
    seed: int = 0,
) -> UserHoldoutSplit:
    """Hold out `test_fraction` of EACH eligible user's interactions for test.

    A user is eligible only if they have >= `min_user_interactions` (you cannot
    hold out a test item from a user with a single interaction without leaving
    the model nothing to learn their vector from). Users below the threshold
    stay entirely in train and are excluded from test scoring — they are the
    cold-start population the fallback handles, not something the CF model is
    graded on.

    The number held out per eligible user is `round(test_fraction * n)`, floored
    at 1 and capped so at least `min_train_interactions` remain in train. Which
    interactions are held out is chosen by a `seed`-seeded RNG, deterministically
    in user-row order. Interaction weights are preserved in both matrices.
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction}")
    if min_train_interactions < 1:
        raise ValueError("min_train_interactions must be >= 1")
    if min_user_interactions < min_train_interactions + 1:
        raise ValueError(
            "min_user_interactions must exceed min_train_interactions so an "
            "eligible user can yield both a train and a test item"
        )

    csr = interactions.tocsr()
    n_users, n_items = csr.shape
    indptr, indices, data = csr.indptr, csr.indices, csr.data
    rng = np.random.default_rng(seed)

    tr_rows: list[int] = []
    tr_cols: list[int] = []
    tr_vals: list[float] = []
    te_rows: list[int] = []
    te_cols: list[int] = []
    te_vals: list[float] = []

    for u in range(n_users):
        start, end = indptr[u], indptr[u + 1]
        cols_u = indices[start:end]
        vals_u = data[start:end]
        n = cols_u.shape[0]

        # Decide how many to hold out (0 => user stays entirely in train).
        if n < min_user_interactions:
            n_test = 0
        else:
            n_test = max(1, int(round(test_fraction * n)))
            n_test = min(n_test, n - min_train_interactions)
            n_test = max(n_test, 0)

        if n_test == 0:
            tr_rows.extend([u] * n)
            tr_cols.extend(cols_u.tolist())
            tr_vals.extend(vals_u.tolist())
            continue

        perm = rng.permutation(n)
        test_local, train_local = perm[:n_test], perm[n_test:]

        te_rows.extend([u] * test_local.size)
        te_cols.extend(cols_u[test_local].tolist())
        te_vals.extend(vals_u[test_local].tolist())

        tr_rows.extend([u] * train_local.size)
        tr_cols.extend(cols_u[train_local].tolist())
        tr_vals.extend(vals_u[train_local].tolist())

    shape = (n_users, n_items)
    train = sp.csr_matrix(
        (np.asarray(tr_vals, dtype=np.float64), (tr_rows, tr_cols)), shape=shape
    )
    test = sp.csr_matrix(
        (np.asarray(te_vals, dtype=np.float64), (te_rows, te_cols)), shape=shape
    )
    return UserHoldoutSplit(
        train=train,
        test=test,
        seed=seed,
        test_fraction=test_fraction,
        min_train_interactions=min_train_interactions,
    )


def _hits_in_top_k(recommended: np.ndarray, relevant: set[int], k: int) -> int:
    """Count how many of the top-k recommended items are in `relevant`."""
    return sum(1 for item in recommended[:k] if int(item) in relevant)


def precision_at_k(recommended: np.ndarray, relevant: set[int], k: int) -> float:
    """Fraction of the top-k recommended items that are relevant (in `relevant`).

    `recommended` is an array of item indices, already ranked best-first, with
    the user's TRAIN items already removed (you never recommend what they've
    already listened to). Divides by k (not by the number returned), so a short
    list padded with misses is penalized.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return _hits_in_top_k(recommended, relevant, k) / k


def recall_at_k(recommended: np.ndarray, relevant: set[int], k: int) -> float:
    """Fraction of the user's relevant (held-out) items that appear in top-k.

    Returns 0.0 if the user has no relevant items (undefined recall).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    return _hits_in_top_k(recommended, relevant, k) / len(relevant)


def ndcg_at_k(recommended: np.ndarray, relevant: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance).

    DCG sums 1/log2(rank+1) over relevant hits in the top-k (rank 1-based);
    normalized by the ideal DCG (all relevant items ranked first, capped at k).
    Returns 0.0 if there are no relevant items.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if int(item) in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(recommended: np.ndarray, relevant: set[int], k: int) -> float:
    """Reciprocal rank of the FIRST relevant item in the top-k (0 if none)."""
    if k <= 0:
        raise ValueError("k must be positive")
    for rank, item in enumerate(recommended[:k], start=1):
        if int(item) in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(recommended: np.ndarray, relevant: set[int], k: int) -> float:
    """AP@k: mean of precision-at-each-hit over the top-k, normalised by min(|rel|, k)."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if int(item) in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def per_user_scores(
    recommended: np.ndarray,
    test: sp.csr_matrix,
    user_rows: np.ndarray,
    k: int,
) -> dict:
    """Per-user metric arrays (not macro-averaged), aligned with `user_rows`.

    Returns {metric: np.ndarray of length len(user_rows)}. The macro average of
    each array equals the corresponding value from `evaluate_recommendations`.
    Used for confidence intervals and paired significance tests (resample users).
    """
    test = test.tocsr()
    user_rows = np.asarray(user_rows)
    p, r, n, mrr, ap = [], [], [], [], []
    for row, u in enumerate(user_rows):
        relevant = set(test.indices[test.indptr[u]:test.indptr[u + 1]].tolist())
        rec = recommended[row]
        p.append(precision_at_k(rec, relevant, k))
        r.append(recall_at_k(rec, relevant, k))
        n.append(ndcg_at_k(rec, relevant, k))
        mrr.append(mrr_at_k(rec, relevant, k))
        ap.append(average_precision_at_k(rec, relevant, k))
    return {
        f"precision@{k}": np.array(p), f"recall@{k}": np.array(r), f"ndcg@{k}": np.array(n),
        f"mrr@{k}": np.array(mrr), f"map@{k}": np.array(ap),
    }


def evaluate_recommendations(
    recommended: np.ndarray,
    test: sp.csr_matrix,
    user_rows: np.ndarray,
    k: int,
) -> dict:
    """Macro-average precision@k / recall@k / NDCG@k over scored users.

    recommended : (len(user_rows), >=k) array of ranked item indices per user,
                  aligned row-for-row with `user_rows`, train items already removed.
    test        : CSR (n_users x n_items) of held-out interactions.
    user_rows   : the user indices being scored (each must have >=1 test item).

    Every user counts equally (macro average), regardless of how many
    interactions they have. This is the frozen definition of "the score".
    """
    test = test.tocsr()
    user_rows = np.asarray(user_rows)
    p_sum = r_sum = n_sum = mrr_sum = ap_sum = 0.0
    for row, u in enumerate(user_rows):
        relevant = set(test.indices[test.indptr[u]:test.indptr[u + 1]].tolist())
        rec = recommended[row]
        p_sum += precision_at_k(rec, relevant, k)
        r_sum += recall_at_k(rec, relevant, k)
        n_sum += ndcg_at_k(rec, relevant, k)
        mrr_sum += mrr_at_k(rec, relevant, k)
        ap_sum += average_precision_at_k(rec, relevant, k)
    m = len(user_rows)
    if m == 0:
        return {f"precision@{k}": 0.0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0,
                f"mrr@{k}": 0.0, f"map@{k}": 0.0, "n_users_scored": 0}
    return {
        f"precision@{k}": p_sum / m,
        f"recall@{k}": r_sum / m,
        f"ndcg@{k}": n_sum / m,
        f"mrr@{k}": mrr_sum / m,
        f"map@{k}": ap_sum / m,
        "n_users_scored": m,
    }
