"""Tests for MMR diversity re-ranking (src/rerank.py)."""
import numpy as np

from src import rerank

# items 10 & 11 share an identical embedding (redundant); item 12 is orthogonal.
ITEMS = np.array([10, 11, 12])
SCORES = np.array([1.0, 0.9, 0.1])
FACTORS = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def test_lambda_one_is_pure_relevance():
    # pure relevance keeps the two highest-scoring items, even though redundant
    assert list(rerank.mmr_rerank(ITEMS, SCORES, FACTORS, k=2, lambda_=1.0)) == [10, 11]


def test_high_diversity_avoids_redundant_item():
    # max diversity picks the top item then the dissimilar one, skipping its twin
    assert list(rerank.mmr_rerank(ITEMS, SCORES, FACTORS, k=2, lambda_=0.0)) == [10, 12]


def test_k_caps_at_candidate_count():
    out = rerank.mmr_rerank(ITEMS, SCORES, FACTORS, k=9, lambda_=0.5)
    assert len(out) == 3 and set(out.tolist()) == {10, 11, 12}


def test_empty_pool():
    assert len(rerank.mmr_rerank(np.array([]), np.array([]), np.zeros((0, 2)), k=5)) == 0
