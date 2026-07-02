"""data_loading.py — load Last.fm and build the user-artist interaction matrix.

Reusable, side-effect-free helpers imported by the EDA notebook, the harness,
and the API. No modeling or metric logic lives here (that's eval_core / als).
Filenames and the raw-data location come from configs/data_config.yaml via
src.utils — nothing here hardcodes a relative path.

The interactions file is `user_artists.dat`: `userID \t artistID \t weight`,
where `weight` is a listening count (implicit feedback).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils import get_logger, load_config, raw_data_dir

logger = get_logger("data_loading")


def _files() -> dict:
    """Filename map from data_config.yaml (data.files)."""
    return load_config("data_config")["data"]["files"]


def load_user_artists(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load the raw interactions as a DataFrame [userID, artistID, weight].

    Parameters
    ----------
    raw_dir : str or Path, optional
        Directory holding the .dat files. If None, resolved from
        LASTFM_RAW_DIR / data_config.yaml via src.utils.raw_data_dir().
    """
    base = Path(raw_dir) if raw_dir is not None else raw_data_dir()
    path = base / _files()["user_artists"]
    if not path.exists():
        raise FileNotFoundError(
            f"Interactions file not found at {path}. Download the HetRec 2011 "
            "Last.fm dataset into data/raw/ (see README.md / data_config.yaml)."
        )
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    logger.info("Loaded %s interactions from %s", f"{len(df):,}", path.name)
    return df


def load_artists(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Return artist metadata [id, name, url, pictureURL] for human-readable output.

    artists.dat is UTF-8 (verified); reading it as latin-1 mojibakes accented
    names (e.g. "Björk" -> "BjÃ¶rk"). errors="replace" guards any stray bad byte.
    """
    base = Path(raw_dir) if raw_dir is not None else raw_data_dir()
    return pd.read_csv(base / _files()["artists"], sep="\t",
                       encoding="utf-8", encoding_errors="replace")


@dataclass(frozen=True)
class InteractionMatrix:
    """A user-artist interaction matrix plus the index maps back to raw IDs.

    matrix:    CSR (n_users x n_items), values = listening counts (weights).
    user_ids:  array; user_ids[row] -> original userID.
    item_ids:  array; item_ids[col] -> original artistID.
    user_pos:  dict original userID -> row index.
    item_pos:  dict original artistID -> col index.
    """
    matrix: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_pos: dict[int, int]
    item_pos: dict[int, int]

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape


def build_interaction_matrix(df: pd.DataFrame) -> InteractionMatrix:
    """Build a CSR (users x artists) interaction matrix from raw interactions.

    IDs are densified to contiguous 0..n indices (raw IDs are sparse/gappy), and
    the maps to recover the originals are kept for output and the API.
    """
    user_ids = np.sort(df["userID"].unique())
    item_ids = np.sort(df["artistID"].unique())
    user_pos = {uid: i for i, uid in enumerate(user_ids)}
    item_pos = {iid: i for i, iid in enumerate(item_ids)}

    rows = df["userID"].map(user_pos).to_numpy()
    cols = df["artistID"].map(item_pos).to_numpy()
    vals = df["weight"].to_numpy(dtype=np.float64)

    matrix = sp.csr_matrix(
        (vals, (rows, cols)), shape=(len(user_ids), len(item_ids))
    )
    return InteractionMatrix(matrix, user_ids, item_ids, user_pos, item_pos)


def load_active_matrix() -> InteractionMatrix:
    """Build/load the interaction matrix for the active dataset (configs/data_config.yaml).

    "lastfm360k" -> the filtered 360K core (cached); "lastfm2k" -> the HetRec set.
    Downstream code (make_split, the API) calls this so a dataset swap is one config line.
    """
    ds = load_config("data_config")["data"].get("active_dataset", "lastfm2k")
    if ds == "lastfm360k":
        from src import data_360k
        return data_360k.load_or_build()
    return build_interaction_matrix(load_user_artists())


def sparsity_report(matrix: sp.csr_matrix) -> dict:
    """Density / sparsity summary of the interaction matrix."""
    n_users, n_items = matrix.shape
    n_cells = n_users * n_items
    nnz = matrix.nnz
    return {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_cells": int(n_cells),
        "n_interactions": int(nnz),
        "density": nnz / n_cells,
        "sparsity": 1.0 - nnz / n_cells,
        "sparsity_pct": 100.0 * (1.0 - nnz / n_cells),
    }


def interactions_per_user(matrix: sp.csr_matrix) -> np.ndarray:
    """Number of distinct artists each user has interacted with (one per row)."""
    return np.asarray((matrix > 0).sum(axis=1)).ravel()


def interactions_per_item(matrix: sp.csr_matrix) -> np.ndarray:
    """Number of distinct users who interacted with each artist (one per col)."""
    return np.asarray((matrix > 0).sum(axis=0)).ravel()
