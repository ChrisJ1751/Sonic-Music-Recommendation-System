"""Toy-example validation of the FROZEN eval_core metrics + split. [Milestone 2]

The point of these tests: before trusting precision@k / recall@k / NDCG@k on
real data, prove they compute what we think on a tiny case whose correct answers
we worked out BY HAND. If these pass, the metrics are trustworthy; if eval_core
is ever edited, these are the tripwire.

Worked toy example:
  relevant (held-out) = {0, 2}; ranked recommendations = [0, 1, 2, 3]; k = 3.
    - precision@3 = (hits in top 3) / 3 = |{0, 2}| / 3 = 2/3
    - recall@3    = (hits in top 3) / |relevant| = 2 / 2 = 1.0
    - NDCG@3:
        DCG  = 1/log2(1+1) + 1/log2(3+1) = 1/1 + 1/2 = 1.5   # hits at ranks 1 and 3
        IDCG = 1/log2(1+1) + 1/log2(2+1) = 1.0 + 0.63093 = 1.63093
        NDCG = 1.5 / 1.63093 = 0.91973

Run:  pytest tests/test_eval_core.py -v
"""
import numpy as np
import scipy.sparse as sp

from src.harness import eval_core

REC = np.array([0, 1, 2, 3])
REL = {0, 2}


def test_precision_at_k_toy():
    assert eval_core.precision_at_k(REC, REL, 3) == 2 / 3


def test_recall_at_k_toy():
    assert eval_core.recall_at_k(REC, REL, 3) == 1.0


def test_ndcg_at_k_toy():
    assert abs(eval_core.ndcg_at_k(REC, REL, 3) - 0.91973) < 1e-4


def test_ndcg_perfect_ranking_is_one():
    # relevant items ranked first => NDCG == 1.0
    assert eval_core.ndcg_at_k(np.array([2, 0, 9, 8]), {0, 2}, 4) == 1.0


def test_metric_edges_no_hits_and_empty_relevant():
    miss = np.array([5, 6, 7])
    assert eval_core.precision_at_k(miss, REL, 3) == 0.0
    assert eval_core.recall_at_k(miss, REL, 3) == 0.0
    assert eval_core.ndcg_at_k(miss, REL, 3) == 0.0
    # no relevant items => recall/ndcg defined as 0.0, not a crash
    assert eval_core.recall_at_k(REC, set(), 3) == 0.0
    assert eval_core.ndcg_at_k(REC, set(), 3) == 0.0


def _toy_matrix() -> sp.csr_matrix:
    """4 users with 10, 4, 1, 2 interactions (user 2 is sub-threshold)."""
    rows, cols, vals = [], [], []

    def add(u, items):
        for c in items:
            rows.append(u)
            cols.append(c)
            vals.append(float(c + 1))  # distinct nonzero weights

    add(0, range(0, 10))
    add(1, range(0, 4))
    add(2, [5])
    add(3, [1, 2])
    return sp.csr_matrix((vals, (rows, cols)), shape=(4, 20))


def test_per_user_split_is_disjoint_and_keeps_train_per_test_user():
    M = _toy_matrix()
    split = eval_core.per_user_train_test_split(
        M, test_fraction=0.2, min_user_interactions=2, seed=0
    )
    tr, te = split.train, split.test

    # disjoint (no interaction in both) and a clean partition of all interactions
    assert tr.multiply(te).nnz == 0
    assert tr.nnz + te.nnz == M.nnz

    tr_counts = np.asarray((tr > 0).sum(axis=1)).ravel()
    te_counts = np.asarray((te > 0).sum(axis=1)).ravel()

    # every user with a test item retains >= 1 train item
    for u in np.where(te_counts > 0)[0]:
        assert tr_counts[u] >= 1

    # sub-threshold user (user 2, 1 interaction) is entirely in train, no test
    assert te_counts[2] == 0
    assert tr_counts[2] == 1

    # holdout sizes: round(0.2*n) floored at 1 -> user0:2, user1:1, user3:1
    assert te_counts[0] == 2
    assert te_counts[1] == 1
    assert te_counts[3] == 1

    # weights are preserved, not overwritten
    assert set(tr.data).union(te.data) == set(M.data)


def test_mrr_and_map_toy():
    # relevant {0,2}, recs [0,1,2,3], k=3
    assert eval_core.mrr_at_k(REC, REL, 3) == 1.0  # first hit at rank 1
    # AP@3 = (1/1 + 2/3) / min(2,3) = 1.6667/2 = 0.8333
    assert abs(eval_core.average_precision_at_k(REC, REL, 3) - 0.83333) < 1e-4
    # MRR rewards a later first-hit less:
    assert eval_core.mrr_at_k(np.array([9, 9, 0]), {0}, 3) == 1 / 3
    assert eval_core.mrr_at_k(np.array([9, 9, 9]), {0}, 3) == 0.0


def test_evaluate_recommendations_macro_average():
    # 2 users, 4 items. test relevance: user0 -> {0,2}, user1 -> {1}.
    test = sp.csr_matrix((np.array([1.0, 1, 1]), (np.array([0, 0, 1]), np.array([0, 2, 1]))),
                         shape=(2, 4))
    recommended = np.array([[0, 1], [3, 1]])  # user0 top2=[0,1]; user1 top2=[3,1]
    out = eval_core.evaluate_recommendations(recommended, test, np.array([0, 1]), k=2)
    # by hand: precision macro = (1/2 + 1/2)/2 = 0.5; recall = (1/2 + 1)/2 = 0.75
    assert abs(out["precision@2"] - 0.5) < 1e-9
    assert abs(out["recall@2"] - 0.75) < 1e-9
    # ndcg: user0 = 1/1.63093 = 0.61315; user1 = (1/log2 3)/1 = 0.63093; macro = 0.62204
    assert abs(out["ndcg@2"] - 0.62204) < 1e-4
    # mrr: user0 first hit rank1 -> 1.0; user1 first hit rank2 -> 0.5; macro 0.75
    assert abs(out["mrr@2"] - 0.75) < 1e-9
    # map: user0 AP=(1/1)/min(2,2)=0.5; user1 AP=(1/2)/min(1,2)=0.5; macro 0.5
    assert abs(out["map@2"] - 0.5) < 1e-9
    assert out["n_users_scored"] == 2

    # per-user arrays: their macro mean must equal the aggregate
    per = eval_core.per_user_scores(recommended, test, np.array([0, 1]), k=2)
    for key in ("precision@2", "recall@2", "ndcg@2", "mrr@2", "map@2"):
        assert len(per[key]) == 2
        assert abs(per[key].mean() - out[key]) < 1e-12


def test_split_is_deterministic_given_seed():
    # larger matrix so a different seed reliably holds out different cells
    rng = np.random.default_rng(0)
    dense = (rng.random((40, 60)) < 0.4).astype(float) * rng.integers(1, 100, (40, 60))
    M = sp.csr_matrix(dense)

    a = eval_core.per_user_train_test_split(M, seed=42)
    b = eval_core.per_user_train_test_split(M, seed=42)
    assert (a.test != b.test).nnz == 0  # same seed => identical split

    c = eval_core.per_user_train_test_split(M, seed=7)
    assert (a.test != c.test).nnz > 0  # different seed => different held-out cells
