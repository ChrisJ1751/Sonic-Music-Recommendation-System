# Concept note — implicit vs explicit feedback (and why it changes everything)

A short note for the project record on why this dataset is *implicit* and what
that forces.

## What we have
Last.fm gives a **listening count** per (user, artist). There is no "user X rated
artist Y 4 stars." A play count is **implicit feedback**: evidence of exposure
and preference *strength*, but with no negative signal — a zero means "hasn't
listened," which could be dislike OR simply never-encountered. We cannot tell
which.

## Why this rules out explicit-feedback methods
- **No RMSE/MAE on counts.** Those treat the number as a graded rating to
  reproduce. But a count of 13,883 vs 11,690 (real values for user 2) is not "this
  artist is 1.2 stars better" — it's heavy-tailed exposure. Optimizing squared
  error chases magnitude, not preference ranking.
- **Zeros aren't negatives.** Explicit methods train only on observed ratings.
  Implicit methods must treat unobserved cells as *weak negatives with low
  confidence*, not missing data.

## What implicit ALS does instead (Hu, Koren, Volinsky 2008)
- Split each observed count into a **preference** `p = 1` (any listening) and a
  **confidence** `c = 1 + alpha * count`. We're confident a high-count cell is a
  true positive; we're weakly confident an unobserved cell is a negative.
- Factorize with those confidences as weights. `alpha` (in `search_config.py`)
  is exactly this confidence-scaling knob.

## What it forces in evaluation
Because the target is a *ranking* of unseen items, the metrics are ranking
metrics — **precision@k / recall@k / NDCG@k** — not error-on-counts. See
[`../specs/eval_design.md`](../specs/eval_design.md).

Recorded as a decision: `decisions.md`, 2026-06-30.
