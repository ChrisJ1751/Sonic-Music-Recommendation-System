"""run_session.py — the disciplined search SESSION (pre-registered, bounded).

This is the autoresearch loop made auditable. Every config is written down here
FIRST, with a one-line reason, BEFORE any results are seen — so the search is a
small set of pre-registered comparisons, not an open roam that would manufacture
a lucky winner at this sparsity (see src/harness/program.md). Each config is
validated against `search_config.BOUNDS` and scored over several seeds by the
frozen runner, which appends to `outputs/experiments/log.jsonl`.

It RANKS and reports. It promotes nothing. A human reads the leaderboard, records
the chosen config in `decisions.md`, and only then confirms once on the locked
holdout (`confirm_holdout.py`).

The plan is anchored on the Milestone-3 ablation (log1p confidence, low alpha
best): an alpha sweep at baseline capacity, a factors sweep at the best alpha, a
regularization sweep, and an iteration-convergence check.

Usage:  python -m src.harness.run_session
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import scipy.sparse as sp  # noqa: E402

from src.harness import run_search  # noqa: E402
from src.utils import TEST_PATH, TRAIN_PATH, get_logger  # noqa: E402

logger = get_logger("harness.run_session")

# --- pre-registered configs (written before looking at results) -------------
PRE_REGISTERED: list[dict] = [
    dict(tag="baseline", note="shipped baseline",
         factors=64, regularization=0.01, iterations=15, alpha=10.0),
    dict(tag="alpha1_f64", note="alpha sweep: low confidence (ablation favourite)",
         factors=64, regularization=0.01, iterations=15, alpha=1.0),
    dict(tag="alpha3_f64", note="alpha sweep: mild confidence",
         factors=64, regularization=0.01, iterations=15, alpha=3.0),
    dict(tag="alpha30_f64", note="alpha sweep: high confidence",
         factors=64, regularization=0.01, iterations=15, alpha=30.0),
    dict(tag="alpha1_f16", note="factors sweep: very low capacity (sparse data favours this)",
         factors=16, regularization=0.01, iterations=15, alpha=1.0),
    dict(tag="alpha1_f24", note="factors sweep: low capacity",
         factors=24, regularization=0.01, iterations=15, alpha=1.0),
    dict(tag="alpha1_f32", note="factors sweep: low capacity",
         factors=32, regularization=0.01, iterations=15, alpha=1.0),
    dict(tag="alpha1_f128", note="factors sweep: higher capacity",
         factors=128, regularization=0.01, iterations=15, alpha=1.0),
    dict(tag="alpha1_f192", note="factors sweep: highest capacity (overfit watch)",
         factors=192, regularization=0.01, iterations=15, alpha=1.0),
    dict(tag="alpha1_f128_reg.001", note="regularization sweep: weaker",
         factors=128, regularization=0.001, iterations=15, alpha=1.0),
    dict(tag="alpha1_f128_reg.1", note="regularization sweep: stronger",
         factors=128, regularization=0.1, iterations=15, alpha=1.0),
    dict(tag="alpha1_f128_iters30", note="iteration check: convergence",
         factors=128, regularization=0.01, iterations=30, alpha=1.0),
]

RULE = "=" * 92


def _fmt(band: dict) -> str:
    return f"{band['mean']:.4f}+/-{band['std']:.4f}"


def print_leaderboard(rows: list[dict], k: int) -> None:
    rows = sorted(rows, key=lambda r: r["metrics"][f"ndcg@{k}"]["mean"], reverse=True)
    print("\n" + RULE)
    print(f"SEARCH SESSION LEADERBOARD  (ranked by NDCG@{k} mean over {rows[0]['n_seeds']} seeds)"
          "  — PROPOSALS, NOT PROMOTIONS")
    print(RULE)
    hdr = (f"{'#':<3}{'tag':<22}{'fac':>4}{'reg':>8}{'it':>4}{'alpha':>7}  "
           f"{'NDCG@'+str(k):<18}{'P@'+str(k):>9}{'R@'+str(k):>9}{'cov':>7}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows, 1):
        m = r["metrics"]
        print(f"{i:<3}{r['tag']:<22}{r['factors']:>4}{r['regularization']:>8.3f}"
              f"{r['iterations']:>4}{r['alpha']:>7.1f}  "
              f"{_fmt(m[f'ndcg@{k}']):<18}{m[f'precision@{k}']['mean']:>9.4f}"
              f"{m[f'recall@{k}']['mean']:>9.4f}{m[f'coverage@{k}']['mean']:>7.3f}")
    print(RULE)
    best = rows[0]
    print(f"Top by NDCG@{k}: {best['tag']}  (factors={best['factors']}, "
          f"reg={best['regularization']}, iters={best['iterations']}, alpha={best['alpha']}).")
    print("This is a PROPOSAL. A human records the choice in decisions.md and then runs")
    print("confirm_holdout.py once. The session promotes nothing automatically.")
    print(RULE)


def main() -> None:
    train = sp.load_npz(TRAIN_PATH)
    test = sp.load_npz(TEST_PATH)
    logger.info("running %d pre-registered configs", len(PRE_REGISTERED))
    rows = []
    for cfg in PRE_REGISTERED:
        rows.append(run_search.run(dict(cfg), train=train, test=test))
    print_leaderboard(rows, k=run_search.DEFAULT_K)


if __name__ == "__main__":
    main()
