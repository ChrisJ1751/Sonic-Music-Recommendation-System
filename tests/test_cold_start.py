"""Tests for the cold-start fallback (src/cold_start.py)."""
import numpy as np
import scipy.sparse as sp

from src import cold_start


def _toy():
    # item 0: 3 listeners, item 1: 1, item 2: 2, item 3: 0 (never listened)
    # counts deliberately != listener counts, to prove we rank by DISTINCT listeners.
    dense = np.array([
        [9, 0, 1, 0],
        [5, 0, 0, 0],
        [1, 7, 1, 0],
    ], dtype=float)
    return sp.csr_matrix(dense)


def test_item_popularity_counts_distinct_listeners():
    pop = cold_start.item_popularity(_toy())
    assert list(pop) == [3, 1, 2, 0]


def test_popular_items_ranks_by_listeners_not_plays():
    # despite item0 having the biggest play counts, ranking is by listeners:
    # item0 (3) > item2 (2) > item1 (1) > item3 (0)
    recs = cold_start.popular_items(_toy(), k=3)
    assert list(recs) == [0, 2, 1]


def test_popular_items_excludes_known_and_stops_at_k():
    recs = cold_start.popular_items(_toy(), k=2, exclude={0})
    assert list(recs) == [2, 1]  # item0 excluded
    assert len(recs) == 2


def test_recommend_cold_start_is_popularity():
    t = _toy()
    assert list(cold_start.recommend_cold_start(t, k=3)) == list(cold_start.popular_items(t, k=3))
