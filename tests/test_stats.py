"""Tests for bootstrap CIs and paired significance tests (src/stats.py)."""
import numpy as np

from src import stats


def test_bootstrap_ci_constant_has_zero_width():
    out = stats.bootstrap_ci(np.full(50, 0.3))
    assert abs(out["mean"] - 0.3) < 1e-12
    assert abs(out["hi"] - out["lo"]) < 1e-9


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.1, size=500)
    out = stats.bootstrap_ci(vals, seed=1)
    assert out["lo"] < out["mean"] < out["hi"]
    assert out["hi"] - out["lo"] < 0.05  # ~tight for n=500


def test_paired_bootstrap_detects_clear_difference():
    a = np.full(300, 0.6)
    b = np.full(300, 0.4)
    out = stats.paired_bootstrap_diff(a, b, seed=0)
    assert abs(out["mean_diff"] - 0.2) < 1e-9
    assert out["significant"] and out["lo"] > 0
    assert out["p_two_sided"] < 0.05


def test_paired_bootstrap_no_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(0.5, 0.1, size=400)
    out = stats.paired_bootstrap_diff(a, a.copy(), seed=0)
    assert abs(out["mean_diff"]) < 1e-12
    assert not out["significant"]
