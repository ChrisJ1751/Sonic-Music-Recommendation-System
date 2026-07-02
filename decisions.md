# Decision Log

Append-only. **Newest at top.** Never delete an entry — supersede it with a new
one that cites the old. Dates are absolute.

Template:

```
## YYYY-MM-DD — <short title>
**Decision:** ...
**Why:** ...
**Revisit when:** ... (optional)
**Supersedes:** <date/title> (optional)
```

---

## 2026-07-01 — Deep-learning capstone: keep EASE served, Mult-VAE is the discovery alternative
**Decision:** After promoting EASE, run one more experiment (`src/exp_deep_360k.py`)
— a properly-trained deep **Mult-VAE** (600/200, 40 epochs) against the full zoo on
the identical frozen 360K split — and **keep EASE as the served model**. Record the
deep model as a legitimate *discovery-oriented* alternative, not the served choice.
**Result (NDCG@10, full-ranking, all p<0.001):** EASE 0.219 > Mult-VAE 0.194 >
ALS 0.184 > item-item BM25 0.110 > popularity 0.044. The Mult-VAE went from *last*
on 2k to *2nd* on 360K (overtakes ALS), but still trails EASE by +0.026. Coverage:
Mult-VAE 0.811 vs EASE 0.419 — ~2x reach at *lower* accuracy.
**Why:** (1) Completes the pivot loop honestly — the deep model that lost on small
data was re-tested on real data, and either outcome was informative. (2) Confirms
the Dacrema/Rendle prior: a well-designed linear model beats a deep one on accuracy
even at scale. (3) Settles the "more coverage → more accuracy?" question with data:
coverage and accuracy rise together only up to EASE, then trade off (accuracy-vs-
discovery frontier); coverage does not buy accuracy.
**Corrections made:** two displayed numbers were stale/misleading and were fixed
everywhere — item-item BM25 on 360K is **0.110** (not the 0.150 estimate), and
EASE catalog coverage is **0.419** measured over the full held-out set (the earlier
"0.199" was a 1,500-user sampling artifact).
**Revisit when:** trying EASE variants (EDLAE), RecVAE, tuned iALS (Rendle 2022),
or hybrid content features — none expected to change the headline.

## 2026-07-01 — Pivot to Last.fm-360K, fix the split, promote EASE as served model
**Decision:** Scale the project from Last.fm-2k (HetRec) to **Last.fm-360K**, fix a
training-data-starvation bug in the split, and switch the served model from ALS
to **EASE**.
**Why (three linked findings):**
1. **The 2k data was capping us.** Each 2k user is truncated to their top-50
   artists — an artificial task, and only ~92k interactions. 360K has real,
   *uncapped* histories (17.6M interactions). We filter to a dense core (artists
   with >=100 listeners, a seeded 40k-user sample): **39,499 users x 11,607
   artists, 1.68M interactions** (`src/data_360k.py`, cached).
2. **The three-way split starved training.** Holdout 20% + test 20% left only
   ~64% to train on, which alone dropped NDCG@10 from 0.23 to 0.17 on 2k. Fixed
   to **holdout 10% / test ~13.5% / train ~76.5%** (`make_split.py`).
3. **On richer data, EASE beats tuned ALS.** On the 2k data low-capacity ALS won
   and deep models lost; on 360K the *linear* EASE autoencoder (Steck 2019) wins:
   **NDCG@10 0.219 vs ALS 0.184**, and Mult-VAE 0.16. This is the classic
   "capacity pays off as data grows" story. EASE is now the served model.
**On the "low NDCG" concern:** 0.219@10 is the harsh-cutoff view of a strong
model under *honest full-ranking* evaluation. At the cutoffs the literature uses,
our EASE scores **NDCG@100=0.361, Recall@50=0.423** — squarely in the published
SOTA band on the comparable Million Song Dataset (EASE ~0.39/0.43, Mult-VAE
~0.32/0.36). We report the full cutoff curve so @10 is never seen alone.
**Project framing:** Phase 1 (2k) = methodology development (frozen harness,
disciplined ALS search, significance, cold-start; notebooks 00-07). Phase 2
(360K) = scaling to real data (notebook 08, the served API/demo). The 2k
notebooks stand as the Phase-1 record.
**Revisit when:** adding track-level (song) recommendations, the agreed next step.

## 2026-06-30 — Notebook depth, significance testing, and an encoding-bug fix
**Decision:** Deepen the analytical notebooks and add statistical rigor.
- **Significance testing**: `src/stats.py` (bootstrap CI + paired user-level
  bootstrap) + `eval_core.per_user_scores`. Result: ALS beats the strong
  item-item BM25 baseline by **+0.0274 NDCG@10, 95% CI [0.021, 0.034],
  p < 0.001** — the test Dacrema et al. say is usually skipped. Notably, ALS
  wins for only 46% of users (loses for 30%), which argues for routing, not a
  blanket choice.
- **Deeper notebooks**: 00 gains data-quality checks + a Lorenz curve / Gini
  (popularity Gini 0.73 by listeners) + the tag landscape; 01 gains a runnable
  demonstration that a user-level split makes CF blind (0/378 held-out users have
  history); 02 gains per-user NDCG distribution + a mainstreamness analysis
  (mainstream-taste users served 3.1x better — an equity finding); 04 gains a 2D
  PCA embedding coloured by genre (clusters by genre with no genre supervision);
  05 gains the significance tests.
- **Bug fixed**: `load_artists` read artists.dat as latin-1, but the file is
  UTF-8 — accented names were mojibake ("Björk" -> "BjÃ¶rk"). Switched to UTF-8;
  added a regression test.
**Why:** The notebooks were too thin to carry the portfolio, and a comparison
without a significance test isn't credible by current standards. The encoding bug
was a visible data-quality defect.

## 2026-06-30 — Hardening pass: strong baselines, beyond-accuracy metrics, MMR, packaging
**Decision:** Extend the project to staff-level standards, grounded in recsys
literature, without changing the chosen model.
- **Benchmarked against strong baselines** (Ferrari Dacrema et al., RecSys 2019,
  "Are We Really Making Much Progress?"): popularity, item-item BM25, BPR, and the
  chosen ALS. On the frozen split: popularity 0.063, BPR 0.119, **item-item BM25
  0.144**, ALS 0.171 (NDCG@10). The honest benchmark is BM25, not popularity; ALS
  wins but by a modest margin over a simple neighbourhood model — stated plainly.
  Code: `src/models.py`, `notebooks/05`.
- **Beyond-accuracy metrics** (Kaminskas & Bridge 2017): added MRR@k, MAP@k to the
  frozen `eval_core`; coverage, novelty (self-information), Gini, intra-list
  diversity to `src/metrics.py`. The search log now records the full suite.
- **MMR diversity re-ranking** (Carbonell & Goldstein 1998): `src/rerank.py`,
  exposed as `diversity` on the API — the concrete lever for the
  accuracy-vs-discovery tradeoff the model card flags.
- **Packaging/CI**: `pyproject.toml` (canonical deps + ruff + pytest config),
  ruff clean, GitHub Actions CI (ruff + pytest on 3.12), Dockerfile for the API.
  API gains a `/similar-artists/{id}` endpoint.
**Why:** A portfolio recommender that doesn't compare against strong baselines or
report beyond-accuracy behaviour is not credible by current standards; the
engineering surface (lint/CI/Docker) is what separates a notebook from a
shippable service.
**Revisit when:** never relitigated; future model work re-runs the comparison.

## 2026-06-30 — Milestone 4: chosen config + locked-holdout confirmation
**Decision:** Promote **factors=32, regularization=0.01, iterations=15, alpha=1.0**
(log1p confidence) as the served config. Recorded in `search_config.CONFIG`
(tag "chosen"); the API serves it.
**Session:** 12 pre-registered configs (`run_session.py`), 3 seeds each, ranked
by NDCG@10 on the search-visible test split. Findings:
- **Low capacity wins.** factors 16/24/32 form a plateau (NDCG 0.169-0.171, bands
  overlap) and quality drops sharply above it (f64=0.157, f128=0.138, f192=0.121).
  Extreme sparsity rewards small models.
- **alpha=1 (the floor) is best**; NDCG falls monotonically as alpha rises.
- regularization (0.001/0.01/0.1 at f128) and iterations (15 vs 30) make no
  material difference — the model is converged and not regularization-limited.
**Why f32 over the nominal top f24:** f24 (0.1712+/-0.0031) and f32
(0.1710+/-0.0006) are statistically tied (overlapping bands), but f32's seed band
is ~5x tighter. Per program.md's acceptance rule (overlapping = no real
difference; prefer the stabler/simpler), f32 is the credible pick. All three
beat popularity (0.063) by ~2.7x.
**Holdout confirmation (one-time, `confirm_holdout.py`, read once):** trained on
the full search pool (74,265 interactions), the chosen config scores
**NDCG@10 = 0.2330 +/- 0.0006** on the locked holdout — higher than CV because of
the extra training data and more held-out items per user. It generalises with no
sign of overfitting.
**Known tradeoff:** the chosen config has low catalog coverage@10 (~0.04). Higher
alpha/factors raise coverage (to ~0.11) at a real NDCG cost — a discovery-vs-
accuracy product call, flagged in the model card, not re-litigated here.
**Revisit when:** a product goal weights catalog coverage/diversity, or a new
data export changes the sparsity regime.

## 2026-06-30 — Milestone 3: log-scale counts before confidence weighting
**Decision:** ALS confidence is `c = 1 + alpha * log(1 + count)` (log1p of the
listen count), not `c = 1 + alpha * count`. Implemented in `als_model.to_confidence`
(default `count_transform="log1p"`; `"linear"` kept for the ablation).
**Why:** Last.fm counts span 1..352,698 (6 orders of magnitude). With raw counts,
a few mega-played artists dominate every user's confidence and ALS degrades as
alpha rises. A pre-search ablation on the search-visible split (factors=64,
reg=0.01, iters=15) showed, at NDCG@10: popularity 0.063; raw-count ALS peaks
0.098 at alpha=1 and falls to 0.024 at alpha=100; **log1p ALS reaches 0.158 at
alpha=1** and stays strong across alpha. Log-scaling is Hu et al.'s own variant
for skewed counts — faithful, not a hack. Full ablation in notebooks/02.
**Revisit when:** never casually; switching the transform changes what `alpha`
means and must supersede this entry.

## 2026-06-30 — Milestone 2: split parameters and three-way partition
**Decision:** `eval_core.py` is implemented and toy-validated (7 tests pass).
The locked holdout is sealed via `make_split.py` as a **three-way per-user
partition**, reusing the frozen split twice:
- **Locked holdout** = 20% per eligible user, `HOLDOUT_SEED=1` (sealed; loop
  never reads it).
- **Search train/test** from the remaining pool = 20% test per user,
  `SPLIT_SEED=0`.
Result: train 59,411 / test 14,854 / holdout 18,569 = 92,834 (reconciles).
Per-user medians 32 / 8 / 10. 8 cold-start users (<2 interactions) stay in train,
excluded from test/holdout scoring.
**Why:** Reusing the one frozen split function for both cuts guarantees the
holdout and the search test obey identical leakage-safe rules. Two seeds keep the
two cuts independent and reproducible.
**Revisit when:** changing any of `HOLDOUT_FRACTION`, `TEST_FRACTION`,
`HOLDOUT_SEED`, `SPLIT_SEED` requires superseding this entry and re-running
`make_split` (it overwrites deterministically).

## 2026-06-30 — Adopt FPD repo conventions: utils path-anchoring + make_split sole-owner
**Decision:** Add `src/utils.py` as the single source of truth for paths
(`PROJECT_ROOT`-anchored), config loading, logging, and hashing — modeled on the
FPD project's `src/utils.py`. All modules and notebooks resolve data through it;
no relative-path or `os.chdir` hacks. The locked-holdout path is defined **only**
in `src/harness/make_split.py` (the one-time human-run split), never in
`eval_core.py`/`run_search.py`, so loop code has no symbol pointing at the
holdout.
**Why:** Reviewed the FPD repo's actual `src/` modules and harness. The
path-anchoring convention makes notebooks/scripts cwd-independent; the
make_split sole-ownership convention is the structural mechanism that enforces
"the loop never reads the holdout" (it's not just a rule, there's no symbol to
call). Matching these now keeps the two projects' discipline consistent.

## 2026-06-30 — Implicit feedback framing; ranking metrics, not RMSE
**Decision:** Treat the Last.fm `weight` (per-user-artist listen count) as
**implicit feedback**. Model with implicit ALS using confidence weighting
`c = 1 + alpha * count`. Evaluate with **precision@k / recall@k / NDCG@k**, never
RMSE/MAE on the raw counts.
**Why:** There are no explicit ratings in this dataset — a play count is a signal
of *exposure/preference strength*, not a graded rating. Counts are also wildly
heavy-tailed (a few mega-listened artists per user). RMSE on counts would chase
magnitude, which is not the recommendation task. Ranking metrics measure what we
actually serve: an ordered list of artists a user hasn't seen yet.
**Revisit when:** never, for this dataset, unless an explicit-rating source is added.

## 2026-06-30 — Evaluation split = per-user interaction holdout, not naive row split
**Decision:** The frozen split (in `src/harness/eval_core.py`) holds out a
fraction of *each user's* interactions for test, keeping that same user's other
interactions in train. It is NOT a random split of the interaction rows, and NOT
a split of users into train/test groups.
**Why:** See `docs/specs/eval_design.md` for the full argument. In short: (a) a
random row split can put a user's only-three artists partly in train and partly
in test, but to recommend *for* a user at test time the model must have learned
that user's vector from *train* interactions — a naive split leaks the test
artists into the user factor and inflates scores; (b) splitting whole users into
a test group makes ALS unable to score them at all (no learned user vector =
cold start), which measures the cold-start fallback, not the CF model. Per-user
holdout is the standard, leakage-safe protocol for top-N recommendation.
**Revisit when:** never (this is the frozen contract); changes require a new
entry and re-validation of `eval_core`.

## 2026-06-30 — Reuse FPD's three-file discipline; do NOT assume its stability layer
**Decision:** Adopt the autoresearch-style three-file architecture from the FPD
project (frozen `eval_core.py` + agent-editable `search_config.py` + human
`program.md`). Do **not** port FPD's stability-across-seeds/folds machinery
unless the EDA demonstrates sparsity creates an analogous evaluation fragility.
**Why:** FPD needed stability machinery because its data was tiny *and severely
imbalanced* (194 positives). This dataset's challenge is **sparsity, not
imbalance**. Copying the stability layer without that justification would be
cargo-culting. The investigation is tracked in
`docs/specs/sparsity_fragility_investigation.md`.
**Revisit when:** the EDA (Milestone 1) is complete — decide then, from the
per-user interaction distribution, whether a stability layer is warranted.

## 2026-06-30 — Project kickoff, dataset, and structure
**Decision:** Use Last.fm HetRec 2011 (`user_artists.dat`). Adopt the repo
structure documented in `README.md` (configs / data / docs / notebooks / src /
api / tests / outputs). Deliverable is a FastAPI JSON endpoint; no frontend.
**Why:** Music-domain dataset directly relevant to a Spotify application; the
structure mirrors the proven FPD template so the discipline transfers.
