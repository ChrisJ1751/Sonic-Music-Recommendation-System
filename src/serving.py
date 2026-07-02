"""Shared inference core for the served recommender (Last.fm-360K + EASE).

Both the FastAPI service (`api/main.py`) and the Streamlit app
(`app/streamlit_app.py`) import from here, so the recommendation logic lives in
exactly one place. Everything is a plain function over an immutable `RecoState`;
the web layers are thin adapters that add HTTP / widgets on top.

Served model: EASE (Steck 2019), the linear item-item autoencoder that won the
model comparison on this data. IDs are matrix indices (user row / artist column)
because the 360K native ids are opaque hashes and names.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402

from src import als_model, cold_start, models, rerank  # noqa: E402
from src.data_loading import load_active_matrix  # noqa: E402
from src.utils import PROCESSED_DIR, get_logger, load_config  # noqa: E402

logger = get_logger("serving")

EASE_REG = 100.0
EASE_B_PATH = PROCESSED_DIR / "lastfm360k" / "ease_B.npy"


@dataclass
class RecoState:
    """Everything needed to serve recommendations, loaded once."""

    dataset: str
    im: object          # InteractionMatrix (matrix + item_ids)
    B: np.ndarray       # EASE item-item weight matrix
    Xbin: object        # binarised CSR interactions (for EASE scoring)
    als: object         # small ALS model, item embeddings for MMR diversity

    @property
    def n_users(self) -> int:
        return int(self.im.matrix.shape[0])

    @property
    def n_artists(self) -> int:
        return int(self.im.matrix.shape[1])


def _fit_or_load_ease(matrix) -> np.ndarray:
    if EASE_B_PATH.exists():
        logger.info("loading cached EASE B from %s", EASE_B_PATH.name)
        return np.load(EASE_B_PATH)
    logger.info("fitting EASE (reg=%.0f) on %d items ...", EASE_REG, matrix.shape[1])
    B = models.fit_ease(matrix, reg=EASE_REG)
    EASE_B_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EASE_B_PATH, B)
    return B


def load_state() -> RecoState:
    """Load the dataset and fit/load the served model. Call once at startup."""
    dataset = load_config("data_config")["data"].get("active_dataset", "lastfm2k")
    im = load_active_matrix()
    B = _fit_or_load_ease(im.matrix)
    als, _ = als_model.train_als(im.matrix, factors=64, regularization=0.01,
                                 iterations=15, alpha=1.0, seed=0)
    Xbin = (im.matrix > 0).astype(np.float32).tocsr()
    logger.info("serving ready: dataset=%s, EASE(reg=%.0f), %d users, %d artists",
                dataset, EASE_REG, im.matrix.shape[0], im.matrix.shape[1])
    return RecoState(dataset=dataset, im=im, B=B, Xbin=Xbin, als=als)


def name(state: RecoState, col: int) -> str:
    return str(state.im.item_ids[int(col)])


def ease_scores(state: RecoState, user_row: int) -> np.ndarray:
    """EASE scores over all items for one user, with already-played items masked."""
    x = state.Xbin.getrow(user_row)
    scores = np.asarray(x @ state.B).ravel()
    scores[x.indices] = -np.inf
    return scores


def is_known_user(state: RecoState, user_id: int) -> bool:
    return 0 <= user_id < state.n_users


def recommend(state: RecoState, user_id: int, k: int = 10, diversity: float = 0.0) -> dict:
    """Top-k recommendations. EASE for known users (with optional MMR diversity),
    popularity fallback for unknown ones. Returns strategy + list of dicts."""
    n_items = state.n_artists
    if is_known_user(state, user_id):
        scores = ease_scores(state, user_id)
        if diversity > 0.0:
            pool = min(max(k * 6, 60), n_items)
            cand = np.argpartition(-scores, pool - 1)[:pool]
            cand = cand[np.argsort(-scores[cand])]
            cols = rerank.mmr_rerank(cand, scores[cand], state.als.item_factors[cand],
                                     k=k, lambda_=1.0 - diversity)
            strategy = "ease+mmr"
        else:
            cols = np.argpartition(-scores, k - 1)[:k]
            cols = cols[np.argsort(-scores[cols])]
            strategy = "ease"
        out_scores = scores[cols]
    else:
        cols = cold_start.recommend_cold_start(state.im.matrix, k=k)
        out_scores = cold_start.item_popularity(state.im.matrix)[cols].astype(float)
        strategy = "cold_start_popularity"

    recs = [{"artist_id": int(c), "name": name(state, c), "score": float(s)}
            for c, s in zip(cols, out_scores, strict=True)]
    return {"user_id": user_id, "strategy": strategy, "k": k, "recommendations": recs}


def user_profile(state: RecoState, user_id: int, k: int = 12) -> dict:
    """A user's most-played artists (their taste profile)."""
    if not is_known_user(state, user_id):
        return {"user_id": user_id, "in_dataset": False, "top_artists": []}
    row = state.im.matrix.getrow(user_id)
    order = np.argsort(row.data)[::-1][:k]
    return {
        "user_id": user_id, "in_dataset": True, "n_artists": int(row.nnz),
        "top_artists": [{"artist_id": int(row.indices[i]), "name": name(state, int(row.indices[i])),
                         "plays": int(row.data[i])} for i in order],
    }


def similar_artists(state: RecoState, artist_id: int, k: int = 10) -> dict | None:
    """'Fans also like' — nearest artists in EASE's symmetrised item-item weights.
    Returns None if the artist_id is out of range."""
    if not (0 <= artist_id < state.n_artists):
        return None
    sim = state.B[artist_id, :] + state.B[:, artist_id]
    sim[artist_id] = -np.inf
    ids = np.argpartition(-sim, k - 1)[:k]
    ids = ids[np.argsort(-sim[ids])]
    similar = [{"artist_id": int(c), "name": name(state, c), "score": float(sim[c])} for c in ids]
    return {"artist_id": artist_id, "name": name(state, artist_id), "k": k, "similar": similar}


def popular_artists(state: RecoState, n: int = 50) -> list[dict]:
    listeners = np.asarray((state.im.matrix > 0).sum(axis=0)).ravel()
    top = np.argsort(listeners)[::-1][:n]
    return [{"artist_id": int(c), "name": name(state, c), "listeners": int(listeners[c])} for c in top]


def sample_users(state: RecoState, n: int = 6, seed: int = 42) -> list[dict]:
    """A deterministic set of users with rich histories, for demo quick-picks."""
    counts = np.asarray((state.im.matrix > 0).sum(axis=1)).ravel()
    eligible = np.where(counts >= np.percentile(counts, 75))[0]
    rows = np.random.default_rng(seed).choice(eligible, size=min(n, len(eligible)), replace=False)
    out = []
    for r in rows:
        row = state.im.matrix.getrow(int(r))
        top = int(row.indices[int(np.argmax(row.data))])
        out.append({"user_id": int(r), "top_artist": name(state, top)})
    return out


# --- static project facts (single source of truth for the "results" surfaces) ---

def about_payload() -> dict:
    """Reported full-ranking results + methodology on Last.fm-360K.

    Kept here so the FastAPI /about endpoint and the Streamlit results page render
    the same numbers (see notebooks/08 and docs/specs/model_card.md).
    """
    return {
        "model": {
            "name": "EASE",
            "long": "Embarrassingly Shallow Autoencoder (Steck, WWW 2019)",
            "kind": "closed-form linear item-item autoencoder",
            "detail": "B = -P / diag(P), where P = (XᵀX + λI)⁻¹; no hidden layers, no SGD.",
        },
        "headline": [
            {"label": "NDCG@10", "value": 0.219, "note": "tightest cutoff"},
            {"label": "NDCG@100", "value": 0.361, "note": "SOTA band"},
            {"label": "Recall@50", "value": 0.423, "note": "matches MSD SOTA"},
            {"label": "MAP@10", "value": 0.112, "note": "full ranking"},
        ],
        # NDCG@10 on one frozen 360K split/seed (src/exp_deep_360k.py).
        "leaderboard": [
            {"model": "EASE", "ndcg10": 0.219, "served": True},
            {"model": "Mult-VAE (deep)", "ndcg10": 0.194, "served": False},
            {"model": "ALS (f128)", "ndcg10": 0.184, "served": False},
            {"model": "item-item BM25", "ndcg10": 0.110, "served": False},
            {"model": "popularity", "ndcg10": 0.044, "served": False},
        ],
        "significance": "EASE beats tuned ALS by +0.036 NDCG@10 (95% CI [0.034, 0.037], p<0.001). "
                        "A properly-trained deep Mult-VAE overtakes ALS on 360K (+0.010, p<0.001) but "
                        "still trails EASE (−0.026, p<0.001): capacity helps, yet the linear model stays on top.",
        # Accuracy vs beyond-accuracy on the full held-out set. Coverage = fraction of the
        # 11,607 artists that appear in some user's top-10 (measured over all scored users).
        "beyond_accuracy": [
            {"model": "EASE", "ndcg10": 0.219, "coverage": 0.419, "novelty": 5.26, "served": True},
            {"model": "Mult-VAE (deep)", "ndcg10": 0.194, "coverage": 0.811, "novelty": 6.09, "served": False},
            {"model": "ALS (f128)", "ndcg10": 0.184, "coverage": 0.188, "novelty": 5.64, "served": False},
            {"model": "item-item BM25", "ndcg10": 0.110, "coverage": 0.091, "novelty": 3.65, "served": False},
            {"model": "popularity", "ndcg10": 0.044, "coverage": 0.002, "novelty": 3.16, "served": False},
        ],
        "cutoff_curve": {
            "k": [10, 20, 50, 100],
            "ndcg": [0.219, 0.284, 0.339, 0.361],
            "recall": [0.194, 0.278, 0.423, 0.531],
            # Best published EASE on the Million Song Dataset (Steck 2019), at the
            # cutoffs where each is reported — so we compare like-for-like, not a
            # @10 point against a @100 line.
            "sota_points": [
                {"metric": "ndcg", "k": 100, "value": 0.389},
                {"metric": "recall", "k": 20, "value": 0.333},
                {"metric": "recall", "k": 50, "value": 0.428},
            ],
        },
        "methodology": [
            {"title": "Frozen evaluation harness",
             "body": "Metrics and the per-user split live in a locked eval_core.py; "
                     "no tuning ever touches the sealed holdout."},
            {"title": "Leakage-safe holdout",
             "body": "Per-user interaction split (train / search-test / locked holdout); "
                     "the holdout was read exactly once."},
            {"title": "Full-catalogue ranking",
             "body": "Every model ranks all 11,607 artists — no sampled-negative "
                     "shortcuts (Krichene & Rendle, KDD 2020)."},
            {"title": "Paired significance tests",
             "body": "5,000-resample paired bootstrap on per-user NDCG, so every "
                     "'X beats Y' claim carries a CI and p-value."},
            {"title": "Strong baselines, not just popularity",
             "body": "Benchmarked against ALS, BPR, item-item BM25 and a deep VAE "
                     "(Ferrari Dacrema et al., RecSys 2019)."},
            {"title": "Reproducible + tested",
             "body": "Deterministic given seeds (BLAS pinned to 1 thread); tests, "
                     "ruff-clean, CI on every push."},
        ],
        "pivot": "Phase 1 on Last.fm-2k: low-capacity ALS won and the deep VAE lost. "
                 "Phase 2 on real, uncapped Last.fm-360K (1.68M interactions): capacity pays off. "
                 "The deep Mult-VAE climbs from last to overtake ALS, and EASE pulls ahead of everything "
                 "to become the served model. Complexity earns its keep only once there is enough data — "
                 "but the linear model still wins.",
        "stack": ["Python", "implicit / PyTorch", "scipy.sparse", "FastAPI", "Streamlit", "pytest"],
    }
