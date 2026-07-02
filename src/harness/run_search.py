"""run_search.py — THE FROZEN RUNNER (the autoresearch loop, our adaptation).

Glues the three files together for one search attempt:
  1. Reads `search_config.CONFIG` (the agent-editable knobs) and validates it
     against `search_config.BOUNDS`.
  2. Loads the search-visible per-user train/test split (train.npz / test.npz).
  3. Trains confidence-weighted implicit ALS on TRAIN and evaluates with the
     frozen `eval_core` metrics on TEST, over several seeds, and reports the
     mean and std (the lightweight seed-sensitivity check the sparsity
     investigation called for — not the full FPD stability apparatus).
  4. Appends ONE row to outputs/experiments/log.jsonl. **Promotes nothing.**

It never reads the locked holdout (it has no symbol for that path) and never
edits a frozen file. Mapping to karpathy/autoresearch:
  prepare.py -> eval_core.py (frozen) ; train.py -> search_config.py (knobs only,
  not arbitrary code) ; program.md -> program.md ; results.tsv -> log.jsonl.
The deliberate inversions: a *validated config*, not free-form code; a *human
gate*, not auto-commit; a *bounded budget*, not run-forever.

Usage:  python -m src.harness.run_search
"""
from __future__ import annotations

import os

# Pin BLAS to one thread BEFORE numpy/implicit load: implicit parallelises over
# users itself, so nested BLAS threads cause contention (and a noisy warning).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib  # noqa: E402
import json  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from src import als_model, metrics
from src.harness import eval_core, search_config
from src.utils import EXPERIMENTS_DIR, TEST_PATH, TRAIN_PATH, get_logger

logger = get_logger("harness.run_search")

LOG_PATH = EXPERIMENTS_DIR / "log.jsonl"
DEFAULT_SEEDS = (0, 1, 2)   # lightweight seed-sensitivity check (see decisions.md)
DEFAULT_K = 10


def validate(cfg: dict, bounds: dict) -> None:
    """Reject any config that strays outside the documented bounds."""
    for key, (lo, hi) in bounds.items():
        if key not in cfg:
            raise ValueError(f"config missing required key: {key!r}")
        if not (lo <= cfg[key] <= hi):
            raise ValueError(
                f"{key}={cfg[key]} out of bounds [{lo}, {hi}] — "
                "changing a bound is a protocol change (decisions.md), not a tuning move."
            )


def config_hash(cfg: dict) -> str:
    payload = {k: cfg[k] for k in sorted(cfg) if k in search_config.BOUNDS}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def evaluate_config(
    cfg: dict,
    seeds=DEFAULT_SEEDS,
    k: int = DEFAULT_K,
    train: sp.csr_matrix | None = None,
    test: sp.csr_matrix | None = None,
) -> dict:
    """Train + score one config across seeds; return metric means/stds.

    `train`/`test` can be passed in (e.g. from a notebook) to avoid re-loading;
    otherwise the search-visible split is loaded from disk.
    """
    if train is None:
        if not TRAIN_PATH.exists():
            raise FileNotFoundError(
                f"{TRAIN_PATH} not found. Build the split first: "
                "python -m src.harness.make_split"
            )
        train = sp.load_npz(TRAIN_PATH)
    if test is None:
        test = sp.load_npz(TEST_PATH)
    train, test = train.tocsr(), test.tocsr()
    scored = np.where(np.asarray((test > 0).sum(axis=1)).ravel() > 0)[0]

    per_seed: list[dict] = []
    for seed in seeds:
        model, conf = als_model.train_als(
            train,
            factors=cfg["factors"],
            regularization=cfg["regularization"],
            iterations=cfg["iterations"],
            alpha=cfg["alpha"],
            seed=seed,
        )
        recs = als_model.recommend_top_n(model, conf, scored, n=k)
        m = eval_core.evaluate_recommendations(recs, test, scored, k)
        m[f"coverage@{k}"] = metrics.catalog_coverage(recs, train.shape[1])
        m["novelty"] = metrics.novelty(recs, train)
        m["gini"] = metrics.gini(recs, train.shape[1])
        per_seed.append(m)

    metric_keys = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"mrr@{k}",
                   f"map@{k}", f"coverage@{k}", "novelty", "gini"]
    agg = {}
    for key in metric_keys:
        vals = np.array([s[key] for s in per_seed], dtype=float)
        agg[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    agg["n_users_scored"] = int(per_seed[0]["n_users_scored"])
    return agg


def append_log(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run(cfg: dict, seeds=DEFAULT_SEEDS, k: int = DEFAULT_K, log: bool = True,
        train: sp.csr_matrix | None = None, test: sp.csr_matrix | None = None) -> dict:
    """Validate, evaluate, log one config. Returns the log row. Promotes nothing.

    `train`/`test` may be passed (e.g. by the session driver) to avoid reloading
    the split for every config.
    """
    validate(cfg, search_config.BOUNDS)
    scores = evaluate_config(cfg, seeds=seeds, k=k, train=train, test=test)
    row = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_hash": config_hash(cfg),
        "tag": cfg.get("tag", ""),
        "note": cfg.get("note", ""),
        "k": k,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        **{key: cfg[key] for key in search_config.BOUNDS},
        "metrics": scores,
        "holdout_read": False,  # audit guard — True is a bug
    }
    if log:
        append_log(row)
    nd = scores[f"ndcg@{k}"]
    logger.info(
        "logged [%s] %s  NDCG@%d=%.4f+/-%.4f  P@%d=%.4f  R@%d=%.4f  cov=%.3f",
        row["config_hash"], cfg.get("tag", ""), k, nd["mean"], nd["std"],
        k, scores[f"precision@{k}"]["mean"], k, scores[f"recall@{k}"]["mean"],
        scores[f"coverage@{k}"]["mean"],
    )
    return row


def main() -> None:
    run(dict(search_config.CONFIG))


if __name__ == "__main__":
    main()
