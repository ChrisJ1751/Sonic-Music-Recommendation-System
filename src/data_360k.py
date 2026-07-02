"""data_360k.py — load and preprocess the Last.fm-360K dataset.

The 360K dataset (Celma) is user_sha1 / artist_mbid / artist_name / plays over
17.6M rows — real, *uncapped* listening histories (unlike the top-50-capped
HetRec 2k set). We reduce it to a dense, recommendable core and cache the result
so downstream code (the same frozen harness) can load it instantly.

Preprocessing (deterministic given seed):
  1. keep artists with >= `min_artist_users` distinct listeners (drops the
     unrecommendable long tail and caps the item count so EASE is feasible),
  2. sample `n_users` users (with enough history) for tractable iteration,
  3. re-filter within the sample so every artist/user still clears a floor.

The item ids ARE artist names (human-readable, handy for the API/demo).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data_loading import InteractionMatrix
from src.utils import PROCESSED_DIR, get_logger, raw_data_dir

logger = get_logger("data_360k")

RAW_SUBDIR = "lastfm-dataset-360K"
PLAYS_FILE = "usersha1-artmbid-artname-plays.tsv"
CACHE_DIR = PROCESSED_DIR / "lastfm360k"
MATRIX_PATH = CACHE_DIR / "matrix.npz"
USERS_PATH = CACHE_DIR / "user_ids.parquet"
ITEMS_PATH = CACHE_DIR / "item_ids.parquet"

# Preprocessing parameters (a change here is a decisions.md entry).
MIN_ARTIST_USERS = 100      # >=100 listeners -> ~16.9k recommendable artists
N_USERS = 40_000            # sampled users, for tractable search/iteration
MIN_USER_ARTISTS = 20       # a user needs a real history to be modelled
SAMPLE_SEED = 0


def _raw_path() -> Path:
    base = raw_data_dir().parent / "raw_360k"
    if not base.exists():
        base = raw_data_dir()
    return base / RAW_SUBDIR / PLAYS_FILE


def build_360k_matrix(min_artist_users: int = MIN_ARTIST_USERS, n_users: int = N_USERS,
                      min_user_artists: int = MIN_USER_ARTISTS, seed: int = SAMPLE_SEED,
                      raw_path=None) -> InteractionMatrix:
    """Build the filtered 360K interaction matrix from the raw TSV."""
    path = raw_path or _raw_path()
    logger.info("reading %s", path)
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 2, 3],
                     names=["user", "artist", "plays"],
                     dtype={"user": "category", "artist": "category", "plays": "float32"},
                     na_filter=False, encoding="utf-8", encoding_errors="replace",
                     quoting=3, on_bad_lines="skip")
    df = df[df["plays"] > 0]

    # 1) keep sufficiently-listened artists
    art_counts = df["artist"].value_counts()
    keep_art = set(art_counts.index[art_counts >= min_artist_users])
    df = df[df["artist"].isin(keep_art)]

    # 2) sample users with enough history
    usr_counts = df["user"].value_counts()
    eligible = usr_counts.index[usr_counts >= min_user_artists].to_numpy()
    rng = np.random.default_rng(seed)
    sampled = set(rng.choice(eligible, size=min(n_users, len(eligible)), replace=False))
    df = df[df["user"].isin(sampled)]

    # 3) re-filter within the sample (artists that lost all their listeners, thin users)
    art2 = df["artist"].value_counts()
    df = df[df["artist"].isin(set(art2.index[art2 >= 20]))]
    usr2 = df["user"].value_counts()
    df = df[df["user"].isin(set(usr2.index[usr2 >= min_user_artists]))]

    u_codes, user_ids = pd.factorize(df["user"].astype(str))
    a_codes, item_ids = pd.factorize(df["artist"].astype(str))
    matrix = sp.csr_matrix((df["plays"].to_numpy(np.float32), (u_codes, a_codes)),
                           shape=(len(user_ids), len(item_ids)))
    user_ids = np.asarray(user_ids)
    item_ids = np.asarray(item_ids)
    logger.info("360K core: %d users x %d artists | %d interactions",
                matrix.shape[0], matrix.shape[1], matrix.nnz)
    return InteractionMatrix(matrix, user_ids, item_ids,
                             {u: i for i, u in enumerate(user_ids)},
                             {a: i for i, a in enumerate(item_ids)})


def save_cache(im: InteractionMatrix) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sp.save_npz(MATRIX_PATH, im.matrix)
    pd.DataFrame({"user": im.user_ids}).to_parquet(USERS_PATH, index=False)
    pd.DataFrame({"artist": im.item_ids}).to_parquet(ITEMS_PATH, index=False)


def load_or_build() -> InteractionMatrix:
    """Return the cached 360K matrix, building and caching it on first use."""
    if MATRIX_PATH.exists():
        matrix = sp.load_npz(MATRIX_PATH).tocsr()
        user_ids = pd.read_parquet(USERS_PATH)["user"].to_numpy()
        item_ids = pd.read_parquet(ITEMS_PATH)["artist"].to_numpy()
        return InteractionMatrix(matrix, user_ids, item_ids,
                                 {u: i for i, u in enumerate(user_ids)},
                                 {a: i for i, a in enumerate(item_ids)})
    im = build_360k_matrix()
    save_cache(im)
    return im


if __name__ == "__main__":
    im = build_360k_matrix()
    save_cache(im)
    print(f"built + cached: {im.shape[0]:,} users x {im.shape[1]:,} artists, {im.matrix.nnz:,} interactions")
