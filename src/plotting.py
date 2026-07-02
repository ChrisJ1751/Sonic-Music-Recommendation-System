"""plotting.py — reusable, portfolio-grade charts for the recommender.

One place for figure styling so every chart in the notebooks and reports looks
consistent and is reproducible from the repo (not from throwaway scripts). The
palette nods to the Spotify target (green accent, near-black text) without being
gaudy.

Each function takes raw arrays, returns (fig, ax), and optionally saves a
high-DPI PNG. Keep the analysis in the notebooks; keep the drawing here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- palette ----------------------------------------------------------------
GREEN = "#1DB954"   # Spotify green — primary bars
DARK = "#191414"    # near-black — text
ACCENT = "#E8743B"  # warm orange — annotations / reference lines
MUTED = "#9AA0A6"   # grey — secondary series
PURPLE = "#8B5CF6"  # violet — a second data series


def set_style() -> None:
    """Apply the project's matplotlib style. Call once per notebook/session."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": DARK,
        "axes.labelcolor": DARK,
        "axes.edgecolor": "#C7C7C7",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.8,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "text.color": DARK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _titles(ax, title: str, subtitle: str | None, xlabel: str, ylabel: str) -> None:
    if subtitle:
        ax.set_title(f"{title}\n", loc="left", pad=18)
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10,
                color=MUTED, ha="left", va="bottom")
    else:
        ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _save(fig, save_path: str | Path | None) -> None:
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)


def plot_interactions_per_user(counts: np.ndarray, save_path=None):
    """Per-user interaction counts on a log y-axis, so the 50-cap spike and the
    tiny cold-start tail are both visible."""
    n_users = counts.size
    at_cap = int((counts == counts.max()).sum())
    cold = int((counts <= 1).sum())
    cap = int(counts.max())

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(counts, bins=np.arange(1, cap + 2) - 0.5, color=GREEN, edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    ax.annotate(f"{at_cap:,} users ({at_cap / n_users:.0%})\nhold exactly {cap}",
                xy=(cap, at_cap), xytext=(cap - 18, at_cap * 0.5),
                fontsize=10, color=DARK, ha="left",
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1))
    ax.annotate(f"only {cold} users have ≤1\n(the cold-start tail)",
                xy=(1, max(cold, 1)), xytext=(6, 30),
                fontsize=10, color=ACCENT, ha="left",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))

    _titles(ax, "Each user is truncated to their top ~50 artists",
            "Per-user interaction counts (log scale). The dataset caps history, so user-side data is uniform.",
            "distinct artists per user", "users (log scale)")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_artist_long_tail(listeners: np.ndarray, save_path=None):
    """Artists-per-listener-count on log-log axes — the classic power-law view.

    `listeners[j]` = number of distinct users who listened to artist j.
    """
    n_artists = listeners.size
    singletons = int((listeners == 1).sum())
    values, freq = np.unique(listeners, return_counts=True)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(values, freq, s=22, color=GREEN, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))

    ax.annotate(f"{singletons:,} artists ({singletons / n_artists:.0%})\nhave exactly 1 listener",
                xy=(1, singletons), xytext=(2.2, singletons * 0.35),
                fontsize=10, color=ACCENT, ha="left",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
    ax.annotate(f"most-listened artist:\n{int(values.max()):,} listeners",
                xy=(values.max(), freq[-1]), xytext=(140, 25),
                fontsize=10, color=DARK, ha="left",
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1))

    _titles(ax, "Artist popularity is a steep long tail",
            "Where the real sparsity lives: most artists have almost no listeners, so CF has little to learn for them.",
            "distinct listeners per artist (log)", "number of artists (log)")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_weight_distribution(weights: np.ndarray, save_path=None):
    """Listen-count distribution on a log-scaled x-axis with real number ticks."""
    med = float(np.median(weights))
    bins = np.logspace(0, np.log10(weights.max()), 50)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(weights, bins=bins, color=GREEN, edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.axvline(med, color=ACCENT, ls="--", lw=1.6)
    ax.text(med * 1.15, ax.get_ylim()[1] * 0.9, f"median {med:,.0f} plays",
            color=ACCENT, fontsize=10, va="top")

    _titles(ax, "Listen counts span six orders of magnitude",
            "From 1 to ~352,700 plays. We treat counts as confidence (c = 1 + alpha*count), not as ratings.",
            "listen count per (user, artist) — log scale", "(user, artist) pairs")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_alpha_curves(alphas, series: dict[str, list[float]], baseline: float | None = None,
                      baseline_label: str = "popularity", ylabel: str = "NDCG@10",
                      title: str = "Confidence scaling: log1p vs raw counts",
                      subtitle: str | None = None, save_path=None):
    """Metric vs alpha (log-x), one line per series, with an optional baseline rule."""
    colors = {"log1p": GREEN, "linear": MUTED, "raw": MUTED}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for label, ys in series.items():
        ax.plot(alphas, ys, marker="o", color=colors.get(label, DARK), label=label, lw=2)
    if baseline is not None:
        ax.axhline(baseline, color=ACCENT, ls="--", lw=1.5)
        ax.text(alphas[0], baseline, f" {baseline_label} {baseline:.3f}", color=ACCENT,
                fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(frameon=False)
    _titles(ax, title, subtitle, "alpha (confidence scaling, log scale)", ylabel)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_accuracy_coverage(records: list[dict], save_path=None):
    """Accuracy (NDCG@10) vs catalog coverage tradeoff; one point per (transform, alpha)."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for transform, color in (("log1p", GREEN), ("linear", MUTED)):
        pts = [r for r in records if r["transform"] == transform]
        if not pts:
            continue
        pts = sorted(pts, key=lambda r: r["coverage"])
        ax.plot([p["coverage"] for p in pts], [p["ndcg"] for p in pts],
                marker="o", color=color, label=transform, lw=1.5, alpha=0.9)
        for p in pts:
            ax.annotate(f"a={p['alpha']:g}", (p["coverage"], p["ndcg"]),
                        fontsize=8, color=DARK, xytext=(3, 3), textcoords="offset points")
    ax.legend(frameon=False)
    _titles(ax, "Accuracy vs catalog coverage tradeoff",
            "Higher alpha pushes more of the long tail into recommendations, at a cost to NDCG.",
            "catalog coverage@10 (fraction of artists ever recommended)", "NDCG@10")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_leaderboard(tags, means, stds, chosen=None, baseline=None,
                     baseline_label="popularity", k: int = 10, save_path=None):
    """Horizontal bar of NDCG@k by config (best on top), chosen bar highlighted."""
    order = np.argsort(means)  # ascending -> best ends up at top of barh
    tags = [tags[i] for i in order]
    means = np.asarray(means)[order]
    stds = np.asarray(stds)[order]
    colors = [GREEN if t == chosen else MUTED for t in tags]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(tags) + 1.6))
    ax.barh(range(len(tags)), means, xerr=stds, color=colors,
            error_kw=dict(ecolor=DARK, lw=1, capsize=2), height=0.7)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags, fontsize=9)
    if baseline is not None:
        ax.axvline(baseline, color=ACCENT, ls="--", lw=1.5)
        ax.text(baseline, len(tags) - 0.4, f" {baseline_label} {baseline:.3f}",
                color=ACCENT, fontsize=9, va="top")
    ax.grid(axis="y", visible=False)
    _titles(ax, "Search leaderboard", "Chosen config in green; bars are 3-seed means with std error bars.",
            f"NDCG@{k}", "")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_model_comparison(names, means, stds, highlight=None, ylabel="NDCG@10",
                          title="Model comparison vs strong baselines",
                          subtitle="Simple item-item BM25 is the baseline to beat (Dacrema et al. 2019).",
                          save_path=None):
    """Vertical bars of a metric by model, the chosen model highlighted."""
    colors = [GREEN if n == highlight else MUTED for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(range(len(names)), means, yerr=stds, color=colors,
           error_kw=dict(ecolor=DARK, lw=1, capsize=3), width=0.65)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.grid(axis="x", visible=False)
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9, color=DARK)
    _titles(ax, title, subtitle, "", ylabel)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_metric_by_param(x, means, stds, xlabel: str, title: str,
                         subtitle: str | None = None, ylabel: str = "NDCG@10",
                         logx: bool = False, save_path=None):
    """Line with a +/-std band: a metric as one hyperparameter varies."""
    x = np.asarray(x, float)
    means = np.asarray(means, float)
    stds = np.asarray(stds, float)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, means, marker="o", color=GREEN, lw=2)
    ax.fill_between(x, means - stds, means + stds, color=GREEN, alpha=0.18)
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    _titles(ax, title, subtitle, xlabel, ylabel)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_train_test_split(train_counts: np.ndarray, test_counts: np.ndarray, save_path=None):
    """Per-user train vs held-out test counts for scored users."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.arange(0, max(train_counts.max(), test_counts.max()) + 2) - 0.5
    ax.hist(train_counts, bins=bins, color=GREEN, alpha=0.85, edgecolor="white",
            linewidth=0.3, label="train items / user")
    ax.hist(test_counts, bins=bins, color=ACCENT, alpha=0.8, edgecolor="white",
            linewidth=0.3, label="held-out test items / user")
    ax.legend(frameon=False)
    _titles(ax, "Per-user holdout keeps ~80% to learn from, tests on the rest",
            f"medians: {np.median(train_counts):.0f} train / {np.median(test_counts):.0f} test per scored user.",
            "interactions per user", "users")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax
