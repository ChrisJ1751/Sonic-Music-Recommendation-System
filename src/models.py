"""models.py — model zoo + comparison driver for benchmarking baselines.

Motivated by Ferrari Dacrema et al., "Are We Really Making Much Progress?"
(RecSys 2019): complex recommenders are routinely beaten by well-chosen simple
baselines, so a credible result reports the chosen model against *strong*
baselines on the *same* evaluation. Here we compare, on the frozen per-user
split with the frozen metrics:

  - popularity          (non-personalised, distinct-listener ranking)
  - item-item BM25       (neighbourhood model on BM25-weighted counts)
  - BPR                  (pairwise-ranking matrix factorization)
  - ALS (log1p, chosen)  (our served config)

Every model exposes the same `(train, user_rows, k, seed) -> recs` signature, so
the comparison is apples-to-apples.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import BM25Recommender, bm25_weight
from threadpoolctl import threadpool_limits

from src import als_model, cold_start, metrics
from src.harness import eval_core, search_config


def _recommend(model, user_items: sp.csr_matrix, user_rows: np.ndarray, k: int) -> np.ndarray:
    user_rows = np.asarray(user_rows)
    with threadpool_limits(1, "blas"):
        ids, _ = model.recommend(user_rows, user_items[user_rows], N=k,
                                 filter_already_liked_items=True)
    return np.asarray(ids)


def popularity_recs(train: sp.csr_matrix, user_rows: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    order = np.argsort(cold_start.item_popularity(train))[::-1].tolist()
    out = []
    for u in user_rows:
        known = set(train.getrow(int(u)).indices.tolist())
        rec = []
        for i in order:
            if i not in known:
                rec.append(i)
                if len(rec) == k:
                    break
        out.append(rec)
    return np.array(out)


def itemitem_bm25_recs(train: sp.csr_matrix, user_rows: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    weighted = bm25_weight(train, K1=100, B=0.8).tocsr()
    model = BM25Recommender(K=100, num_threads=1)
    with threadpool_limits(1, "blas"):
        model.fit(weighted, show_progress=False)
    return _recommend(model, weighted, user_rows, k)


def bpr_recs(train: sp.csr_matrix, user_rows: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    model = BayesianPersonalizedRanking(factors=32, regularization=0.01,
                                        iterations=100, random_state=seed)
    fit_mat = train.tocsr().astype(np.float32)
    with threadpool_limits(1, "blas"):
        model.fit(fit_mat, show_progress=False)
    return _recommend(model, fit_mat, user_rows, k)


def als_recs(train: sp.csr_matrix, user_rows: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    cfg = search_config.CONFIG
    model, conf = als_model.train_als(train, factors=cfg["factors"],
                                      regularization=cfg["regularization"],
                                      iterations=cfg["iterations"], alpha=cfg["alpha"], seed=seed)
    return _recommend(model, conf, user_rows, k)


def fit_ease(train: sp.csr_matrix, reg: float = 100.0) -> np.ndarray:
    """EASE (Steck, WWW 2019) item-item weight matrix B.

    Closed form of B = argmin ||X - XB||^2 + reg||B||^2 s.t. diag(B)=0:
    B = -P / diag(P) with P = (X^T X + reg*I)^-1, off-diagonal zeroed. Deterministic.
    O(n_items^3) — a few GB and ~a minute at ~12-17k items.
    """
    X = (train > 0).astype(np.float32).tocsr()
    gram = np.asarray((X.T @ X).todense(), dtype=np.float32)
    diag = np.diag_indices_from(gram)
    gram[diag] += reg
    inv = np.linalg.inv(gram)
    B = inv / (-np.diag(inv))
    B[diag] = 0.0
    return B


def ease_recommend(B: np.ndarray, train: sp.csr_matrix, user_rows: np.ndarray, k: int) -> np.ndarray:
    """Top-k item indices per user from a fitted EASE B, train items masked out."""
    X = (train > 0).astype(np.float32).tocsr()
    user_rows = np.asarray(user_rows)
    out = np.empty((len(user_rows), k), dtype=np.int64)
    for s in range(0, len(user_rows), 1024):
        rows = user_rows[s:s + 1024]
        xr = X[rows]
        scores = np.asarray(xr @ B)
        scores[xr.nonzero()] = -np.inf
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        ri = np.arange(len(rows))[:, None]
        out[s:s + len(rows)] = part[ri, np.argsort(-scores[ri, part], axis=1)]
    return out


def ease_recs(train: sp.csr_matrix, user_rows: np.ndarray, k: int, seed: int = 0,
              reg: float = 100.0) -> np.ndarray:
    """Convenience: fit EASE and recommend in one call (for the comparison harness)."""
    return ease_recommend(fit_ease(train, reg), train, user_rows, k)


# registry, ordered simplest -> most complex
MODELS = {
    "popularity": popularity_recs,
    "item-item BM25": itemitem_bm25_recs,
    "BPR": bpr_recs,
    "ALS log1p (chosen)": als_recs,
}


def compare(train: sp.csr_matrix, test: sp.csr_matrix, k: int = 10,
            seeds=(0, 1, 2)) -> list[dict]:
    """Evaluate every model in MODELS over `seeds`; return aggregated rows."""
    train, test = train.tocsr(), test.tocsr()
    scored = np.where(np.asarray((test > 0).sum(axis=1)).ravel() > 0)[0]
    n_items = train.shape[1]
    keys = [f"ndcg@{k}", f"map@{k}", f"recall@{k}", f"precision@{k}", "coverage", "novelty", "gini"]

    results = []
    for name, fn in MODELS.items():
        per_seed = {key: [] for key in keys}
        seeds_used = seeds if fn not in (popularity_recs,) else seeds[:1]  # popularity is deterministic
        for seed in seeds_used:
            recs = fn(train, scored, k, seed)
            m = eval_core.evaluate_recommendations(recs, test, scored, k)
            per_seed[f"ndcg@{k}"].append(m[f"ndcg@{k}"])
            per_seed[f"map@{k}"].append(m[f"map@{k}"])
            per_seed[f"recall@{k}"].append(m[f"recall@{k}"])
            per_seed[f"precision@{k}"].append(m[f"precision@{k}"])
            per_seed["coverage"].append(metrics.catalog_coverage(recs, n_items))
            per_seed["novelty"].append(metrics.novelty(recs, train))
            per_seed["gini"].append(metrics.gini(recs, n_items))
        row = {"model": name}
        for key in keys:
            vals = np.array(per_seed[key], float)
            row[key] = float(vals.mean())
            row[key + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        results.append(row)
    return results
