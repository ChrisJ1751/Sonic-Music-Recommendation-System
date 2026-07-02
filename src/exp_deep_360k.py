"""Phase-2 capstone experiment: does the deep model win on real data?

Reruns the notebook-08 comparison on the Last.fm-360K per-user split and adds a
properly-trained Mult-VAE, measuring BOTH accuracy (NDCG/MAP/Recall) and
beyond-accuracy (catalog coverage, novelty) for every model, plus paired
bootstrap significance vs the served EASE. This closes the 2k -> 360K pivot loop:
on 2k the deep VAE lost; here we find out what happens with 18x more data.

Unlike the serving path, this ALLOWS multi-threading (deep training would be
unusably slow pinned to one BLAS thread). Run:  python -m src.exp_deep_360k
"""
from __future__ import annotations

import json
import os
import time

_THREADS = str(os.cpu_count() or 4)
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(_THREADS))

from src import als_model, data_360k, deep, metrics, models, stats  # noqa: E402
from src.harness import eval_core  # noqa: E402
from src.utils import PROJECT_ROOT, get_logger  # noqa: E402

logger = get_logger("exp_deep_360k")
OUT = PROJECT_ROOT / "outputs" / "experiments" / "deep_360k_results.json"


def _rel_sets(te, scored):
    te = te.tocsr()
    return [set(te.indices[te.indptr[u]:te.indptr[u + 1]].tolist()) for u in scored]


def evaluate(name, recs100, te, tr, scored, rels, n_items, cutoffs=True):
    top10 = recs100[:, :10]
    agg = eval_core.evaluate_recommendations(top10, te, scored, 10)
    row = {
        "ndcg@10": round(agg["ndcg@10"], 4),
        "map@10": round(agg["map@10"], 4),
        "recall@10": round(agg["recall@10"], 4),
        "coverage": round(metrics.catalog_coverage(top10, n_items), 4),
        "novelty": round(metrics.novelty(top10, tr), 3),
        "gini": round(metrics.gini(top10, n_items), 4),
    }
    if cutoffs:
        cc = {}
        for kk in (10, 20, 50, 100):
            n = r = 0.0
            for i in range(len(scored)):
                n += eval_core.ndcg_at_k(recs100[i], rels[i], kk)
                r += eval_core.recall_at_k(recs100[i], rels[i], kk)
            cc[kk] = [round(n / len(scored), 4), round(r / len(scored), 4)]
        row["cutoffs"] = cc
    pu = eval_core.per_user_scores(top10, te, scored, 10)["ndcg@10"]
    logger.info("%-14s NDCG@10=%.4f coverage=%.4f novelty=%.3f", name, row["ndcg@10"],
                row["coverage"], row["novelty"])
    return row, pu


def main() -> None:
    logger.info("threads=%s", _THREADS)
    im = data_360k.load_or_build()
    split = eval_core.per_user_train_test_split(im.matrix, test_fraction=0.2, seed=0)
    tr, te = split.train.tocsr(), split.test.tocsr()
    scored = np.where(np.asarray((te > 0).sum(axis=1)).ravel() > 0)[0]
    n_items = im.matrix.shape[1]
    rels = _rel_sets(te, scored)
    logger.info("360K split: %d users x %d artists | scored %d", im.matrix.shape[0], n_items, len(scored))

    results, peruser = {}, {}

    t = time.time()
    B = models.fit_ease(tr, reg=100.0)
    logger.info("EASE fit in %.0fs", time.time() - t)
    results["EASE"], peruser["EASE"] = evaluate(
        "EASE", models.ease_recommend(B, tr, scored, 100), te, tr, scored, rels, n_items)

    m, c = als_model.train_als(tr, factors=128, regularization=0.01, iterations=15, alpha=1.0, seed=0)
    results["ALS (f128)"], peruser["ALS (f128)"] = evaluate(
        "ALS (f128)", als_model.recommend_top_n(m, c, scored, 100), te, tr, scored, rels, n_items,
        cutoffs=False)

    t = time.time()
    mvae = deep.train_multvae(tr, epochs=40, batch_size=512, hidden=600, latent=200, seed=0)
    logger.info("Mult-VAE trained in %.0fs", time.time() - t)
    results["Mult-VAE"], peruser["Mult-VAE"] = evaluate(
        "Mult-VAE", deep.multvae_recs(mvae, tr, scored, 100), te, tr, scored, rels, n_items)

    results["popularity"], peruser["popularity"] = evaluate(
        "popularity", models.popularity_recs(tr, scored, 100), te, tr, scored, rels, n_items,
        cutoffs=False)

    sig = {}
    for a, b in [("EASE", "Mult-VAE"), ("Mult-VAE", "ALS (f128)"), ("EASE", "ALS (f128)")]:
        d = stats.paired_bootstrap_diff(peruser[a], peruser[b], seed=0)
        sig[f"{a} - {b}"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
        logger.info("%s - %s = %+.4f (p=%.4f, significant=%s)", a, b, d["mean_diff"],
                    d["p_two_sided"], d["significant"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"results": results, "significance": sig,
                               "config": {"mvae_epochs": 40, "hidden": 600, "latent": 200,
                                          "split": "per_user test_fraction=0.2 seed=0"}}, indent=2))
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
