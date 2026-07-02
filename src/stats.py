"""stats.py — bootstrap confidence intervals and paired significance tests.

Per Ferrari Dacrema et al. (RecSys 2019), reporting that model A beats model B is
only credible with a significance test on the *same* users. We resample users
(the unit of evaluation) with replacement: a percentile bootstrap for a single
metric's CI, and a paired bootstrap for the per-user difference between two
models (paired because both models are scored on the identical user set).
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, ci: float = 95.0,
                 seed: int = 0) -> dict:
    """Percentile bootstrap CI for the mean of per-user `values`."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                          ci: float = 95.0, seed: int = 0) -> dict:
    """Paired bootstrap on per-user difference (a - b), same users for both.

    Returns the mean difference, its CI, and a two-sided bootstrap p-value
    (the proportion of resamples on the far side of zero, doubled). A CI that
    excludes 0 means the difference is significant at the chosen level.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must be paired (same length / same users)")
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    frac_le0 = float((boot <= 0).mean())
    p_two_sided = 2.0 * min(frac_le0, 1.0 - frac_le0)
    return {
        "mean_diff": float(diff.mean()),
        "lo": float(lo), "hi": float(hi),
        "p_two_sided": p_two_sided,
        "significant": bool(lo > 0 or hi < 0),
    }
