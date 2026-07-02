"""confirm_holdout.py — the ONE-TIME, human-gated locked-holdout confirmation.

This is the only loop-external place (besides make_split) that touches the
locked holdout, and it is run by a human exactly once, after a config has been
chosen from the search leaderboard and recorded in decisions.md. It answers a
single question: does the chosen config generalise to interactions sealed before
the search began?

Method (mirrors deployment):
  - Train ALS (the chosen `search_config.CONFIG`) on the full search pool
    = train + test (everything the model is allowed to see).
  - Recommend for each user with locked-holdout interactions, filtering items
    they already have in the search pool.
  - Score those recommendations against their LOCKED holdout items with the
    frozen `eval_core` metrics, over the same seeds, and report mean +/- std.

Reading the holdout is a deliberate, recorded act — print the result and add it
to decisions.md. Do not fold this into the search loop.

Usage:  python -m src.harness.confirm_holdout
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from src import als_model, metrics  # noqa: E402
from src.harness import eval_core, search_config  # noqa: E402
from src.harness.make_split import LOCKED_HOLDOUT_PATH  # only confirmer imports this  # noqa: E402
from src.harness.run_search import DEFAULT_K, DEFAULT_SEEDS  # noqa: E402
from src.utils import TEST_PATH, TRAIN_PATH, get_logger  # noqa: E402

logger = get_logger("harness.confirm_holdout")
RULE = "=" * 78


def main(seeds=DEFAULT_SEEDS, k: int = DEFAULT_K) -> None:
    cfg = dict(search_config.CONFIG)
    train = sp.load_npz(TRAIN_PATH).tocsr()
    test = sp.load_npz(TEST_PATH).tocsr()
    holdout = sp.load_npz(LOCKED_HOLDOUT_PATH).tocsr()

    search_pool = (train + test).tocsr()  # everything the model may learn from
    scored = np.where(np.asarray((holdout > 0).sum(axis=1)).ravel() > 0)[0]

    print(RULE)
    print("LOCKED-HOLDOUT CONFIRMATION (one-time, human-gated)")
    print(RULE)
    print(f"chosen config: factors={cfg['factors']}, reg={cfg['regularization']}, "
          f"iters={cfg['iterations']}, alpha={cfg['alpha']}")
    print(f"train on search pool ({search_pool.nnz:,} interactions); "
          f"score on {len(scored):,} users with holdout items")

    rows = []
    for seed in seeds:
        model, conf = als_model.train_als(
            search_pool, factors=cfg["factors"], regularization=cfg["regularization"],
            iterations=cfg["iterations"], alpha=cfg["alpha"], seed=seed,
        )
        recs = als_model.recommend_top_n(model, conf, scored, n=k)
        m = eval_core.evaluate_recommendations(recs, holdout, scored, k)
        m[f"coverage@{k}"] = metrics.catalog_coverage(recs, search_pool.shape[1])
        rows.append(m)

    def band(key):
        v = np.array([r[key] for r in rows], float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

    print("-" * len(RULE))
    for key in (f"ndcg@{k}", f"precision@{k}", f"recall@{k}", f"coverage@{k}"):
        mean, std = band(key)
        print(f"  holdout {key:<14} {mean:.4f} +/- {std:.4f}")
    print(RULE)
    print("Record this number in decisions.md. The holdout has now been read once.")
    print(RULE)


if __name__ == "__main__":
    main()
