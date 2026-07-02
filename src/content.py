"""content.py — content-based artist similarity from social tags (TF-IDF).

The collaborative model is blind to artists with almost no co-listening — and
~61% of artists have a single listener (notebook 00). Content-based similarity
sidesteps that: it represents each artist by the *tags* users applied to it
(`user_taggedartists.dat`), TF-IDF weighted, and finds neighbours by cosine. It
needs no co-listening, so it works for cold, long-tail artists where CF fails.

This is the "content" half of the content/popularity cold-start story. TF-IDF is
computed directly with scipy (no scikit-learn dependency).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils import get_logger, load_config, raw_data_dir

logger = get_logger("content")


def load_tag_assignments(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Return tag assignments as [artistID, tagID] from user_taggedartists.dat."""
    base = Path(raw_dir) if raw_dir is not None else raw_data_dir()
    fname = load_config("data_config")["data"]["files"]["user_taggedartists"]
    df = pd.read_csv(base / fname, sep="\t", encoding="latin-1")
    return df[["artistID", "tagID"]]


def build_artist_tag_tfidf(item_ids: np.ndarray, item_pos: dict[int, int],
                           raw_dir: str | Path | None = None) -> sp.csr_matrix:
    """Build an L2-normalised TF-IDF (n_items x n_tags) matrix aligned to the
    interaction-matrix item columns. Artists with no tags get a zero row.
    """
    df = load_tag_assignments(raw_dir)
    df = df[df["artistID"].isin(item_pos)]
    rows = df["artistID"].map(item_pos).to_numpy()
    cols = df["tagID"].to_numpy()
    n_items = len(item_ids)
    n_tags = int(cols.max()) + 1 if len(cols) else 1

    counts = sp.csr_matrix((np.ones(len(df)), (rows, cols)), shape=(n_items, n_tags))
    # idf over artists (documents); +1 smoothing
    doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(n_items / (1.0 + doc_freq)) + 1.0
    tfidf = counts.multiply(idf).tocsr()
    # L2-normalise rows so cosine similarity is a plain dot product
    norms = np.sqrt(np.asarray(tfidf.multiply(tfidf).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    tfidf = sp.diags(1.0 / norms) @ tfidf
    logger.info("built artist-tag TF-IDF: %d artists x %d tags (%d tagged)",
                n_items, n_tags, int((norms != 1.0).sum()))
    return tfidf.tocsr()


def content_similar(tfidf: sp.csr_matrix, item_row: int, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Top-k tag-similar artists to `item_row` by cosine (excludes the query).

    Returns (item indices, similarity scores). Empty if the query has no tags.
    """
    row = tfidf.getrow(int(item_row))
    if row.nnz == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    sims = np.asarray((tfidf @ row.T).todense()).ravel()
    sims[int(item_row)] = -np.inf
    top = np.argsort(sims)[::-1][:k]
    top = top[sims[top] > 0]  # drop zero-similarity (no shared tags)
    return top, sims[top]
