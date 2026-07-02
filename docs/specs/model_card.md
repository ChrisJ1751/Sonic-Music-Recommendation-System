# Model Card — Last.fm Artist Recommender

A short, honest description of the served model: what it is, how it makes
recommendations, how it is evaluated, and where it falls short. Written for a
reviewer who did not build it.

> **Currently served (Phase 2, Last.fm-360K): EASE** — the linear item-item
> autoencoder (Steck 2019), fit on the 39,499-user / 11,607-artist core. It won
> the model comparison on the richer 360K data, significantly beating tuned ALS
> (paired bootstrap, p<0.001). Honest full-ranking scores: **NDCG@10 0.22,
> NDCG@100 0.36, Recall@50 0.42** — in the published SOTA band for music
> recommendation (cf. EASE/Mult-VAE on the Million Song Dataset; see
> `notebooks/08`). The sections below document the Phase-1 methodology on the 2k
> data, where low-capacity **ALS** was the chosen model — the discipline that
> carried over unchanged when the dataset scaled.

## Model details

- **Task:** top-N artist recommendation from implicit feedback (listen counts).
- **Approach:** collaborative filtering by matrix factorization — implicit ALS
  (Hu, Koren, Volinsky 2008) via the `implicit` library.
- **Confidence:** `c = 1 + alpha * log(1 + count)`. Counts are log-scaled because
  Last.fm play counts span ~6 orders of magnitude; see `decisions.md` and
  `notebooks/02`.
- **Served configuration** (chosen in Milestone 4, `decisions.md` 2026-06-30):
  `factors=32, regularization=0.01, iterations=15, alpha=1.0`.
- **Code:** `src/als_model.py` (model), `src/harness/eval_core.py` (frozen
  metrics + split), `api/main.py` (serving).

## Intended use

- Offline / batch top-N artist recommendations for users present in the training
  snapshot; a demonstration of a disciplined, auditable recommender pipeline.
- **Out of scope:** real-time/online learning, production traffic, demographic
  targeting, any high-stakes decision. The dataset is a 2011 research snapshot,
  not current listening behaviour.

## Training data

- **Last.fm HetRec 2011** (GroupLens): 1,892 users, 17,632 artists, 92,834
  user–artist listen records. Each user is truncated to their ~top-50 artists, so
  the matrix is 99.72% sparse by construction (`notebooks/00`).
- **Split** (`src/harness/make_split.py`): a three-way per-user interaction
  holdout — 60% train / 15% search-test / 20% locked holdout (per-user fractions;
  reconciles to 92,834). The locked holdout is sealed and was read exactly once.

## How a recommendation is generated

1. Look up the user's row in the trained interaction matrix.
2. ALS scores every artist by the dot product of the user and artist latent
   vectors.
3. Artists the user has already listened to (their train interactions) are
   removed.
4. The top-k remaining artists, highest score first, are returned.

Users absent from the training snapshot (or with no usable history) are served by
the **cold-start fallback** (`src/cold_start.py`): the most popular artists by
*distinct listener count* (robust to the play-count heavy tail). The EDA showed
this cold population is tiny here (~8 users), so the fallback is mainly the API's
zero-interaction safety net.

## Evaluation

- **Protocol:** per-user interaction holdout (no leakage; see
  `docs/specs/eval_design.md`), macro-averaged over scored users, 3 seeds.
- **Metrics:** NDCG@10 (primary), precision@10, recall@10, MRR@10, MAP@10, plus
  beyond-accuracy reporting (coverage, novelty, Gini, intra-list diversity).

**Against strong baselines** (per Ferrari Dacrema et al., RecSys 2019 — compare
against strong simple methods, not just popularity), on the search-visible split:

| model | NDCG@10 | MAP@10 | recall@10 | coverage | novelty (bits) |
|---|---|---|---|---|---|
| popularity | 0.063 | 0.027 | 0.063 | 0.001 | 2.75 |
| BPR | 0.119 | 0.054 | 0.113 | 0.049 | 5.40 |
| item-item BM25 | 0.144 | 0.068 | 0.136 | 0.029 | 3.46 |
| **ALS log1p (served)** | **0.171** | **0.081** | **0.167** | 0.038 | 4.45 |

ALS wins, but the honest benchmark is the simple item-item BM25 model (0.144),
not popularity. A **paired user-level bootstrap** confirms the lead is real:
ALS − BM25 = **+0.0274 NDCG@10, 95% CI [0.021, 0.034], p < 0.001** (`notebooks/05`,
`src/stats.py`). The lead is not uniform, though — ALS wins for ~46% of users and
loses for ~30%, which argues for routing/ensembling rather than a blanket choice.
BPR recommends more novel artists at lower accuracy (an accuracy/novelty frontier).

On the **locked holdout** (read once, trained on the full pool) the served config
scores **NDCG@10 = 0.233 +/- 0.001** — higher than CV because of the extra
training data and more held-out items per user; it generalizes with no overfitting.

### Phase 2 — Last.fm-360K (the served regime)

On the real, uncapped 360K data the model ranking changes and **EASE** becomes the
served model. All models below ran through the *same* frozen split
(`test_fraction=0.2, seed=0`), full-catalogue ranking, measured for accuracy *and*
beyond-accuracy (`src/exp_deep_360k.py`, `notebooks/08`):

| model | NDCG@10 | MAP@10 | recall@10 | coverage | novelty (bits) |
|---|---|---|---|---|---|
| **EASE (served)** | **0.219** | **0.112** | **0.194** | 0.419 | 5.26 |
| Mult-VAE (deep) | 0.194 | 0.094 | 0.178 | 0.811 | 6.09 |
| ALS (128 factors) | 0.184 | 0.089 | 0.163 | 0.188 | 5.64 |
| item-item BM25 | 0.110 | 0.047 | 0.102 | 0.091 | 3.65 |
| popularity | 0.044 | 0.017 | 0.039 | 0.002 | 3.16 |

Every gap is significant (paired user-level bootstrap, all **p < 0.001**):
EASE − Mult-VAE = **+0.026**, Mult-VAE − ALS = **+0.010**, EASE − ALS = **+0.036**.
The deep Mult-VAE — which *lost* to ALS on 2k — climbs to 2nd on 360K (capacity
pays off with data), but still trails EASE. It reaches ~2x the catalogue (0.81 vs
0.42 coverage) at *lower* top-10 accuracy: the **accuracy-vs-discovery frontier**,
so more coverage does not buy accuracy. EASE sits at the sweet spot — best accuracy
with healthy coverage — and is served; the `diversity` (MMR) lever exposes the
frontier at runtime. Under multi-cutoff evaluation EASE reaches **NDCG@100 = 0.36,
Recall@50 = 0.42**, in the published SOTA band (cf. EASE on the Million Song Dataset).

## Limitations and known tradeoffs

- **Long-tail items are effectively unrecommendable.** ~61% of artists have a
  single listener; CF has nothing to learn for them. This caps recall and is a
  property of the data, not a bug.
- **Low catalog coverage.** The served config recommends from ~4% of the
  catalogue. Higher `alpha`/`factors` raise coverage to ~11% but cost ~15–30%
  NDCG — an **accuracy vs discovery** product decision, deliberately left to the
  product owner (`notebooks/02`, `notebooks/03`). The API exposes a `diversity`
  parameter (MMR re-ranking, `src/rerank.py`) as the runtime lever for this.
- **Popularity bias / feedback loops.** Like all CF, the model favours
  already-popular artists and, if its outputs fed back into training, would
  reinforce that. Not mitigated here (offline, single snapshot).
- **No temporal signal.** The snapshot has no usable timestamps for the listen
  matrix; the model cannot capture trend or recency.
- **Stale data.** 2011 listening behaviour; not representative of today.

## Fairness / ethics

- No demographic attributes are used or available; the model cannot directly
  discriminate on protected classes. Indirect popularity bias (above) is the main
  equity concern — it under-serves niche tastes and small artists.

## Reproducibility

Deterministic given seeds (BLAS pinned to one thread). To reproduce end to end:

```bash
python -m src.data_360k                  # build the Last.fm-360K core (Phase 2, served)
python -m src.harness.make_split         # build the sealed split (seeds recorded)
python -m src.harness.run_session        # run the pre-registered search -> log.jsonl
python -m src.harness.confirm_holdout    # one-time holdout confirmation
python -m src.exp_deep_360k              # Phase-2 model gauntlet incl. the deep Mult-VAE
pytest                                   # full suite incl. toy-validated metrics + API
```

Notebooks `00`–`03` reproduce the Phase-1 EDA, harness validation, ALS ablation,
and search readout on the 2k data; `07` is the deep-vs-simple comparison and `08`
is the Phase-2 scale-up to 360K where EASE is chosen and calibrated against SOTA.
