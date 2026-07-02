# Results Summary — Last.fm Music Recommender

One page, for a reader who wants the outcome without the code. Technical detail is
in the [model card](../../docs/specs/model_card.md), the notebooks (`00`–`08`), and
the Quarto report (`report/`).

## What was built

A collaborative-filtering music recommender, built and evaluated the way a research
team would ship one: a **frozen** evaluation harness, a leakage-safe holdout, a
gauntlet of strong baselines, and paired significance tests. The served model is
**EASE** (a linear item-item autoencoder). It is presented three ways from one
shared inference core — a Streamlit app, a FastAPI service, and a written Quarto
report — and unknown users fall back to a popularity recommender.

## The two-phase story

**Phase 1 — Last.fm-2k (methodology).** On the small HetRec set, a disciplined
hyperparameter search found that *low capacity wins*: a 32-factor ALS beat both a
linear EASE and a deep Mult-VAE. That is the classic small-data result — complexity
without enough signal does not pay.

**Phase 2 — Last.fm-360K (the served regime).** Scaling to real, uncapped histories
(39,499 users x 11,607 artists, 1.68M interactions) *flipped* the ranking. Measured
on one frozen split, full-catalogue ranking (NDCG@10, all differences p < 0.001):

| Model | NDCG@10 | Catalog coverage |
|-------|:-------:|:----------------:|
| **EASE (served)** | **0.219** | 0.42 |
| Mult-VAE (deep) | 0.194 | 0.81 |
| ALS (128 factors) | 0.184 | 0.19 |
| item-item BM25 | 0.110 | 0.09 |
| popularity | 0.044 | 0.00 |

EASE wins on accuracy. Under multi-cutoff evaluation it reaches **NDCG@100 = 0.36,
Recall@50 = 0.42** — in the published state-of-the-art band (cf. EASE on the Million
Song Dataset).

## What the analysis established

1. **Capacity pays off with data — but the linear model still wins.** The deep
   Mult-VAE climbed from *last* on 2k to *2nd* on 360K, overtaking tuned ALS
   (+0.010, p < 0.001), yet still trailed EASE (−0.026, p < 0.001).
2. **Coverage does not buy accuracy.** Mult-VAE reaches ~2x EASE's catalogue (0.81
   vs 0.42) at *lower* top-10 accuracy — the accuracy-vs-discovery frontier.
   Coverage and accuracy rise together only up to EASE, then trade off. The API's
   `diversity` (MMR) lever exposes that frontier at runtime rather than hard-coding
   a single point.
3. **The small-data set was a trap.** Last.fm-2k capped every user at ~50 artists
   and left 61% of artists with a single listener — an artificial shape that
   depressed scores and hid the true model ranking. Real data was necessary.

## How to trust it

The split and metrics are frozen and validated on a hand-worked toy example; the
search is a small set of pre-registered configs, seed-checked and logged; the final
Phase-1 number came from a holdout read exactly once; every "X beats Y" claim on
360K carries a paired user-level bootstrap CI and p-value. 53 tests pass, ruff-clean.
The full decision trail is in `decisions.md`; the deep-model gauntlet is
`src/exp_deep_360k.py`.

## Honest limitations

2011 data; popularity bias inherent to CF; long-tail artists under-served; no
temporal signal; pure collaborative filtering (no content features). See the model
card for the full list. Next planned: track-level (song) recommendations on a more
recent dataset (e.g. the Spotify Million Playlist Dataset).
