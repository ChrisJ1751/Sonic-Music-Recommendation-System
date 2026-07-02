"""make_split.py — ONE-TIME split script. NOT loop code — run by a human.

It is the ONLY module that knows the locked-holdout path. The search loop
(`run_search.py`) and the metric/split definitions (`eval_core.py`) have no
symbol pointing here, so loop code cannot read the holdout.

What it does (three-way per-user partition, reusing the FROZEN split twice):
  1. Build the full user-artist interaction matrix (src.data_loading).
  2. Seal a LOCKED holdout: hold out HOLDOUT_FRACTION of each eligible user's
     interactions (seed HOLDOUT_SEED). Written under _LOCKED_holdout/ with a
     DO_NOT_READ.md. The loop never reads this.
  3. From the remaining search pool, make the search-visible TRAIN/TEST split
     (seed SPLIT_SEED) that run_search will read.
  4. Save the index maps (matrix row/col -> original userID/artistID) and print
     a reconciliation + cold-start summary.

Every partition reuses eval_core.per_user_train_test_split, so the holdout and
the search split obey the exact same leakage-safe, per-user rules. Re-running
overwrites deterministically (seeds are fixed). Reading the locked holdout is a
one-time, human-gated promotion step logged in decisions.md — never the loop.

Usage:  python -m src.harness.make_split
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data_loading import load_active_matrix
from src.harness import eval_core
from src.utils import (
    HARNESS_DIR,
    ITEM_INDEX_PATH,
    TEST_PATH,
    TRAIN_PATH,
    USER_INDEX_PATH,
    get_logger,
)

logger = get_logger("harness.make_split")

# --- Locked holdout location (defined ONLY here; never in utils/run_search) --
LOCKED_DIR = HARNESS_DIR / "_LOCKED_holdout"
LOCKED_HOLDOUT_PATH = LOCKED_DIR / "holdout_LOCKED.npz"
LOCKED_README_PATH = LOCKED_DIR / "DO_NOT_READ.md"

# --- Split parameters (changing any of these is a decisions.md entry) -------
# Train on ~80% of each user's history: the old 0.2/0.2 three-way starved
# training (only ~64% left to learn from) and depressed scores — see decisions.md
# 2026-07-01. Now: holdout 10%, test ~13.5%, train ~76.5%.
HOLDOUT_FRACTION = 0.1    # per-user fraction sealed into the locked holdout
TEST_FRACTION = 0.15     # per-user fraction of the remaining ~90% pool used for search test
HOLDOUT_SEED = 1         # seals the locked holdout
SPLIT_SEED = 0           # search-visible train/test

RULE = "=" * 78


def _per_user_counts(m: sp.csr_matrix) -> np.ndarray:
    return np.asarray((m > 0).sum(axis=1)).ravel()


def main() -> None:
    print(RULE)
    print("MUSIC RECOMMENDER — ONE-TIME SPLIT (frozen correctness layer)")
    print(RULE)

    im = load_active_matrix()
    full = im.matrix
    n_users, n_items = full.shape

    # 1) Seal the locked holdout: test = the sealed slice, train = the search pool.
    sealed = eval_core.per_user_train_test_split(
        full, test_fraction=HOLDOUT_FRACTION, seed=HOLDOUT_SEED
    )
    locked_holdout = sealed.test
    search_pool = sealed.train

    # 2) Search-visible train/test from the pool.
    split = eval_core.per_user_train_test_split(
        search_pool, test_fraction=TEST_FRACTION, seed=SPLIT_SEED
    )
    train, test = split.train, split.test

    # --- invariants: clean three-way partition of all interactions ----------
    assert train.multiply(test).nnz == 0, "train/test overlap"
    assert train.multiply(locked_holdout).nnz == 0, "train/holdout overlap"
    assert test.multiply(locked_holdout).nnz == 0, "test/holdout overlap"
    assert train.nnz + test.nnz + locked_holdout.nnz == full.nnz, "partition lost/gained interactions"

    # --- write artifacts -----------------------------------------------------
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    LOCKED_DIR.mkdir(parents=True, exist_ok=True)
    sp.save_npz(TRAIN_PATH, train)
    sp.save_npz(TEST_PATH, test)
    sp.save_npz(LOCKED_HOLDOUT_PATH, locked_holdout)
    pd.DataFrame({"row": np.arange(n_users), "userID": im.user_ids}).to_parquet(USER_INDEX_PATH, index=False)
    pd.DataFrame({"col": np.arange(n_items), "artistID": im.item_ids}).to_parquet(ITEM_INDEX_PATH, index=False)
    LOCKED_README_PATH.write_text(
        "# LOCKED HOLDOUT — DO NOT READ FROM THE SEARCH LOOP\n\n"
        f"Sealed {locked_holdout.nnz:,} interactions ({HOLDOUT_FRACTION:.0%} per eligible user, "
        f"seed {HOLDOUT_SEED}).\n"
        "This is the single, human-gated arbiter. The loop has no code path or symbol\n"
        "pointing here (eval_core.py and run_search.py do not define this path).\n"
        "Reading it is a one-time promotion step decided by a human, logged in decisions.md.\n",
        encoding="utf-8",
    )

    # --- summary -------------------------------------------------------------
    full_counts = _per_user_counts(full)
    n_coldstart = int((full_counts < 2).sum())

    print("\n[1] INTERACTION MATRIX")
    print(f"    {n_users:,} users x {n_items:,} artists | {full.nnz:,} interactions")
    print(f"    cold-start users (<2 interactions, excluded from test/holdout): {n_coldstart}")

    print("\n[2] THREE-WAY PER-USER PARTITION (leakage-safe, same frozen rule each time)")
    print(f"    {'split':<22}{'interactions':>14}{'users scored':>14}{'median/user':>13}")
    for name, m in (("train (search-visible)", train), ("test (search-visible)", test),
                    ("holdout (LOCKED)", locked_holdout)):
        c = _per_user_counts(m)
        scored = int((c > 0).sum())
        med = int(np.median(c[c > 0])) if scored else 0
        print(f"    {name:<22}{m.nnz:>14,}{scored:>14,}{med:>13}")

    print("\n[3] RECONCILIATION")
    print(f"    train {train.nnz:,} + test {test.nnz:,} + holdout {locked_holdout.nnz:,} "
          f"= {train.nnz + test.nnz + locked_holdout.nnz:,}  (full {full.nnz:,})  OK")
    print(f"    seeds: holdout={HOLDOUT_SEED}, search split={SPLIT_SEED}  (deterministic)")
    print(f"\n    search-visible -> {TRAIN_PATH.name}, {TEST_PATH.name} under {HARNESS_DIR}")
    print(f"    LOCKED holdout -> {LOCKED_HOLDOUT_PATH}  (+ DO_NOT_READ.md)")
    print(RULE)
    print("Split complete. run_search reads only train.npz / test.npz.")
    print(RULE)


if __name__ == "__main__":
    main()
