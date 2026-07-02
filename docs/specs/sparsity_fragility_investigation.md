# Sparsity & Fragility Investigation

> The question Chris asked explicitly: *don't assume this needs FPD's
> stability-across-seeds machinery — investigate whether sparsity here creates
> an analogous fragility, and say so honestly.* This is that investigation.
> Numbers from `outputs/reports/eda_stats.json` (Milestone 1, 2026-06-30).

## TL;DR / verdict

- **Overall sparsity: 99.72%** (92,834 interactions in a 1,892 × 17,632 =
  33.4M-cell matrix). High, but **structural, not organic** — see below.
- **User-side cold start is NOT a real concern.** Interactions per user are
  effectively **capped at 50** (median 50, mean 49.07, max 50; 1,829 / 1,892
  users have exactly 50). Only **8 users have ≤1** interaction and **21 have
  ≤10**. The dataset is each user's *top-~50 artists* by construction (GroupLens
  readme: "avg. 49.067 artists most listened by each user").
- **Item-side sparsity is the real story.** Median artist has **1** listener;
  **10,679 of 17,632 artists (61%) have exactly one listener**, 74% have ≤2.
  This long tail is what's genuinely hard to recommend and what caps achievable
  recall.
- **FPD-style fragility: only a mild, different version applies.** A
  **lightweight seed-sensitivity check** is justified; the **full FPD
  stability-across-folds/vintage layer is NOT.** Reasoning below.

## What "sparsity" means here vs. what it sounds like

99.72% sparse *sounds* like "users barely interact, so we can't learn them."
That's the FPD-style intuition and it's **wrong here**. The matrix is sparse
because each user is **truncated to their top ~50 artists** out of 17,632 — not
because users are data-poor. Every modeled user brings a full, dense-enough
50-item taste profile. So:

- The per-user holdout split (`eval_core.py`) is **well-conditioned on the user
  axis**: hold out 20% of 50 = 10 test items, keep 40 to learn the user vector,
  for essentially every user uniformly. No user is too thin to split (the 8
  single-interaction users are excluded and routed to cold-start).
- The sparsity bites on the **item axis**: a CF model has almost nothing to
  learn for an artist only one person listened to. Those artists are nearly
  unrecommendable and, when they land in a user's test set, are near-impossible
  to retrieve.

## Is this fragile the way FPD was? Compare the mechanisms

| | FPD (needed the stability layer) | This dataset |
|---|---|---|
| Core problem | tiny **and severely imbalanced** (194 positives, 2.43%) | extreme **sparsity**, no imbalance |
| What the metric is averaged over | a handful of positives → PR-AUC swings wildly on resample | ~1,829 users × ~10 held-out items ≈ **18k test points**, macro-averaged |
| Dominant variance source | which few positives land in val | which items land in a user's test, weighted by the item long tail |
| Temporal/regime structure | yes (2023→2024 base-rate shift) → vintage folds | **none** — no time dimension in the modeled signal |

The evaluation here rests on **orders of magnitude more test points** than FPD's
194 positives, macro-averaged over ~1,829 users. The law of large numbers makes
the headline metric far more stable than FPD's. There is **no temporal regime**
to block folds across. So the two heaviest pieces of FPD's machinery —
many-seed CI bands to tame a 194-positive metric, and vintage-blocked CV — have
**no analogous justification** here.

**The one real residual fragility:** *which* of a user's 50 artists get held out
is random, and because so many artists are rare (1-listener), a test fold that
happens to hold out more long-tail artists will show lower recall than one that
holds out popular ones. That's a genuine seed effect — but it's a **modest
variance** issue on a large average, not the existential one FPD had.

## Decision (proposed; confirm before Milestone 2)

1. **Build a lightweight seed check, not the FPD stability layer.** Run the
   per-user split under ~3–5 seeds and report the **mean and std** of NDCG@10 /
   precision@10 / recall@10. If std is small (expected, given ~18k test points),
   a single locked split is fine for the search; if surprisingly large,
   reconsider. This is cheap and earns the right to use one split.
2. **Treat the item long tail as a modeling reality, not an eval bug.** Expect
   absolute recall to be capped by the ~61% single-listener artists. Report a
   **popularity baseline** — at this item skew it will be a strong, honest
   reference point, and beating it meaningfully is the bar.
3. **Cold-start fallback is for items/edge users, sized small on the user side.**
   Only ~8–21 users are thin; the fallback's main value is a sensible default
   and the API's zero-interaction path, not rescuing a large cold population.

> This is the honest answer to "does sparsity here create an FPD-style
> fragility?": **partially, and mildly.** We adopt the *discipline* (frozen eval,
> pre-registered search, no auto-promote) but not the *stability apparatus*,
> because this dataset's numbers don't call for it. Recorded for Milestone 2
> sign-off.

## Postscript — the prediction held (Milestones 3–4)

The lightweight 3-seed check was built into the runner. Across the entire
12-config search (`outputs/experiments/log.jsonl`), the NDCG@10 seed standard
deviation ranged ~0.0003–0.0031 against means of 0.12–0.17 — i.e. seed noise is
~0.2–2% of signal, and config differences are far larger than the bands. The
macro-averaged metric is stable exactly as predicted, so the single locked split
plus a 3-seed check was sufficient; the FPD many-seed/vintage apparatus would have
added cost for no information. The one place the small band *mattered* was the
f24-vs-f32 tie, where f32's 5x-tighter band broke the tie — the seed check earned
its keep without being scaled up. Prediction confirmed.
