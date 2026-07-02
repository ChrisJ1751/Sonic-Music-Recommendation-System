"""Regenerate the three report/app figures that told their story poorly.

- cutoff_calibration: our EASE curve vs published SOTA, with SOTA references placed
  at the cutoffs where they are actually published (so @10 is not compared to a
  @100 line). Numbers: our EASE from notebook 08; SOTA = EASE on the Million Song
  Dataset (Steck, WWW 2019: Recall@20 .333, Recall@50 .428, NDCG@100 .389).
- ranking_flip: the 2k -> 360K NDCG@10 slope per model — the pivot in one chart.
- beyond_accuracy_360k: EASE vs a popularity baseline on coverage and novelty,
  measured live on 1,500 users (seed 0).

Run:  python -m src.report_figures
Writes to outputs/figures/ and report/figures/.
"""
from __future__ import annotations

import shutil

import matplotlib.pyplot as plt

from src import plotting as viz
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "figures"
REPORT = PROJECT_ROOT / "report" / "figures"


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    shutil.copy(path, REPORT / name)
    plt.close(fig)
    print(f"wrote {name}")


def cutoff_calibration() -> None:
    ks = [10, 20, 50, 100]
    ndcg = [0.219, 0.284, 0.339, 0.361]
    recall = [0.194, 0.278, 0.423, 0.531]
    # Published EASE-on-MSD SOTA reference values (Steck 2019), at their reported cutoff.
    sota = [("NDCG", 100, 0.389, viz.GREEN), ("Recall", 50, 0.428, viz.PURPLE),
            ("Recall", 20, 0.333, viz.PURPLE)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, ndcg, "o-", color=viz.GREEN, lw=2.5, ms=8, label="our EASE — NDCG@k")
    ax.plot(ks, recall, "s-", color=viz.PURPLE, lw=2.5, ms=8, label="our EASE — Recall@k")

    for metric, k, y, color in sota:
        ax.axhline(y, ls=(0, (2, 3)), color=color, lw=1.6, alpha=0.7)
        ax.annotate(f"SOTA {metric}@{k} = {y:.2f}", (108, y), color=color, fontsize=9,
                    xytext=(4, 4), textcoords="offset points", ha="left", va="bottom")

    ax.set(title="Calibration: our EASE vs published SOTA (dotted = SOTA at its reported cutoff)",
           xlabel="cutoff k", ylabel="metric value", xlim=(8, 128), ylim=(0.15, 0.57))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "cutoff_calibration.png")


def ranking_flip() -> None:
    # NDCG@10, Phase 1 (2k) -> Phase 2 (360K).
    # name, ndcg@10 on 2k, on 360K, colour, lw, left-label dy, right-label dy (pts)
    models = [
        ("EASE", 0.169, 0.219, viz.GREEN, 3.0, -9, 0),
        ("Mult-VAE (deep)", 0.158, 0.194, viz.PURPLE, 2.0, 0, +8),
        ("ALS", 0.171, 0.184, viz.DARK, 2.0, +11, -8),
        ("popularity", 0.063, 0.044, viz.MUTED, 1.5, 0, 0),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x0, x1 = 0, 1
    for name, a, b, color, lw, ldy, rdy in models:
        ax.plot([x0, x1], [a, b], "-", color=color, lw=lw, alpha=0.9, zorder=2)
        ax.scatter([x0, x1], [a, b], color=color, s=70, zorder=3)
        ax.annotate(name, (x0, a), color=color, fontsize=10.5, fontweight="bold",
                    xytext=(-12, ldy), textcoords="offset points", ha="right", va="center")
        ax.annotate(f"{b:.3f}", (x1, b), color=color, fontsize=10.5, fontweight="bold",
                    xytext=(12, rdy), textcoords="offset points", ha="left", va="center")

    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Last.fm-2k\n(small, capped)", "Last.fm-360K\n(real, uncapped)"], fontsize=11)
    ax.set(title="Capacity pays off on real data: both EASE and the deep VAE overtake ALS",
           ylabel="NDCG@10", xlim=(-0.6, 1.4), ylim=(0.02, 0.235))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "ranking_flip.png")


def accuracy_vs_coverage() -> None:
    # NDCG@10 vs catalog coverage on the full 360K held-out set (src/exp_deep_360k.py).
    pts = [
        ("EASE (served)", 0.419, 0.219, viz.GREEN, True),
        ("Mult-VAE (deep)", 0.811, 0.194, viz.PURPLE, False),
        ("ALS", 0.188, 0.184, viz.DARK, False),
        ("item-item BM25", 0.091, 0.110, viz.MUTED, False),
        ("popularity", 0.002, 0.044, viz.MUTED, False),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.annotate("", xy=(0.811, 0.194), xytext=(0.419, 0.219),
                arrowprops=dict(arrowstyle="<->", color=viz.MUTED, ls="--", lw=1.3))
    ax.text(0.62, 0.213, "accuracy ↔ coverage trade-off", color=viz.MUTED, fontsize=9.5, ha="center")
    for name, cov, ndcg, color, served in pts:
        ax.scatter(cov, ndcg, s=230 if served else 120, color=color, zorder=3,
                   edgecolor="white", linewidth=1.6 if served else 0)
        ax.annotate(name, (cov, ndcg), color=color, fontsize=10.5, fontweight="bold",
                    xytext=(9, 7), textcoords="offset points")
    ax.set(title="Accuracy vs coverage on 360K: EASE ranks best, the deep model reaches widest",
           xlabel="catalog coverage — fraction of the 11,607 artists ever recommended",
           ylabel="NDCG@10", xlim=(-0.03, 0.92), ylim=(0.02, 0.245))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "accuracy_vs_coverage.png")


def main() -> None:
    viz.set_style()
    cutoff_calibration()
    ranking_flip()
    accuracy_vs_coverage()


if __name__ == "__main__":
    main()
