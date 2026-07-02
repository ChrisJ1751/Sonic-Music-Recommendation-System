# Evaluation Design — why the split is per-user, and what the metrics mean

> Frozen contract. Implemented in `src/harness/eval_core.py`, validated by
> `tests/test_eval_core.py`. This document is the *why* the code is the *what*.

## The task we are actually evaluating

Top-N recommendation: for a user we already know something about, produce a
ranked list of artists they **haven't listened to yet** that they *would* listen
to. So a correct evaluation must:

1. Let the model learn each test user's taste from **some** of their
   interactions (their "training" history), and
2. Score it on **other**, held-out interactions from the *same* user that the
   model never saw.

Everything below follows from those two requirements.

## Why a naive random row split leaks (and measures the wrong thing)

A "naive row split" shuffles the ~92,834 `(user, artist, weight)` rows and puts,
say, 80% in train and 20% in test. Two things go wrong:

### 1. It leaks — the model sees the answer while forming the user vector
In matrix factorization (ALS), a user's recommendations come from that user's
**latent vector**, which is fit from *that user's rows in the training matrix*.
If a random split leaves some of a user's interactions in train and some in
test, the model fits the user's vector partly toward artists that also appear in
their test set (because tastes are internally correlated — the held-out artists
look like the kept ones). The user vector is therefore **pulled toward the very
items we then "test" on.** Worse, the standard top-N evaluation removes a user's
*train* items from their candidate list — but a naively split user can have the
same artist's signal bleed across the split. The reported precision/NDCG comes
out inflated and does not reflect held-out generalization. This is the classic
"information leak" in recommender evaluation.

### 2. With a different naive split — by user — ALS literally can't score them
The other naive instinct is to split *users* into a train group and a test
group. But ALS has **no latent vector** for a user who contributed zero rows to
the training matrix (that's the cold-start problem). Such a user can only be
served by the popularity/content fallback. So a user-level split doesn't measure
the collaborative-filtering model at all — it measures the fallback. Useful for
evaluating cold start specifically; wrong as the *main* CF metric.

## The correct protocol: per-user interaction holdout

For **each** user with enough interactions, randomly hold out a fraction
(default 20%) of *their* interactions into TEST, keeping the rest in TRAIN.

- The model fits user and item vectors on TRAIN only.
- At evaluation, we rank all items for the user, **remove their TRAIN items**
  (you never recommend what they've already played), and check how many of their
  **held-out TEST items** land in the top-k.
- Train and test are disjoint; every test user keeps ≥1 train interaction, so a
  user vector always exists.

This is leakage-safe (the held-out items never touched the fitted vectors) and
it measures the real task (rank unseen items for a known user).

### Eligibility / the cold-start boundary
A user needs at least `min_user_interactions` (default 2) to be split: you
cannot hold out a test item from a user with a single interaction without
leaving the model nothing to learn their vector from. Users below the threshold
stay entirely in train and are **excluded from CF test scoring** — they are
precisely the cold-start population the fallback owns. The EDA quantifies how
many users this is (see `sparsity_fragility_investigation.md`).

### Determinism
The split takes a `seed`. The locked holdout is produced once with a recorded
seed and sealed (`data/processed/harness/_LOCKED_holdout/`); the search uses a
separate train/test split and never reads the locked holdout.

## The metrics (all at cutoff k, default k=10)

Let `relevant` = the user's held-out TEST items, and `recommended` = the model's
ranked top-k after removing TRAIN items.

- **precision@k** = (# of top-k that are relevant) / k. "Of what we showed, how
  much was right." Penalizes filling the list with misses.
- **recall@k** = (# of top-k that are relevant) / |relevant|. "Of what they'd
  have liked, how much did we surface." Note its ceiling is < 1 when
  |relevant| > k.
- **NDCG@k** = DCG / IDCG, binary relevance. DCG = Σ over relevant hits in top-k
  of `1 / log2(rank + 1)` (rank 1-based); IDCG = the DCG of the ideal ordering
  (all relevant items first). Rewards putting hits **higher** in the list, which
  precision/recall ignore. **Primary metric** for the search.

These are averaged over all scored users (macro average), so every user counts
equally regardless of how many interactions they have.

## Worked toy example (the test asserts these exact numbers)

relevant = {A, C}; ranked recommendations = [A, B, C, D]; k = 3.

- precision@3 = 2/3
- recall@3 = 2/2 = 1.0
- NDCG@3: DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5; IDCG = 1/log2(2) +
  1/log2(3) = 1.0 + 0.6309 = 1.6309; NDCG = **0.9197**.

See `tests/test_eval_core.py`.
