# Milestone Plan

Each milestone produces a shareable artifact, so a pause never wastes a phase.

## Milestone 1 — EDA & sparsity (complete)
**Goal:** Quantify the interaction matrix: sparsity (% zero) and the
distribution of interactions per user/artist, to decide whether cold-start and
evaluation-fragility are real concerns here.
**Artifacts:**
- `src/data_loading.py` (matrix builder + sparsity helpers)
- `notebooks/00_eda_interaction_matrix.ipynb`
- `outputs/figures/` (interactions-per-user/artist distributions)
- `docs/specs/sparsity_fragility_investigation.md` — the honest call on whether
  sparsity creates an FPD-style fragility that warrants a stability layer.
**Exit:** We can state sparsity %, median interactions/user, the size of the
few-interaction (cold-start-ish) tail, and a decision on the stability layer.

## Milestone 2 — Frozen evaluation layer (complete)
**Goal:** Implement `eval_core.py` (per-user split + precision@k/recall@k/NDCG@k)
and validate against the toy example.
**Artifacts:** `eval_core.py` (implemented), `tests/test_eval_core.py`
(7 passing), `make_split.py` (implemented), sealed locked holdout +
`DO_NOT_READ.md`, `train.npz` / `test.npz` / index maps,
`notebooks/01_evaluation_harness.ipynb` (guided correctness demonstration).
**Exit:** Metrics provably correct on the hand-worked toy case (NDCG@3 = 0.91973
etc.); three-way partition reconciles (59,411 / 14,854 / 18,569 = 92,834) and is
disjoint + leakage-safe. Done 2026-06-30.

## Milestone 3 — ALS + the autoresearch loop wired up (complete)
**Goal:** `als_model.py` implemented; `run_search.py` (the autoresearch *runner*)
runs one config end-to-end — reads `search_config.py`, trains ALS on `train.npz`,
evaluates with the frozen `eval_core` metrics on `test.npz`, appends to
`log.jsonl`. This is where the karpathy three-file mechanism becomes live: the
loop edits only `search_config.py`, runs the frozen runner, never touches
`eval_core.py` or the locked holdout.
**Artifacts:** `als_model.py` (log1p confidence ALS), `run_search.py` (multi-seed
eval + coverage + JSONL log), `eval_core.evaluate_recommendations` (frozen,
toy-tested), `notebooks/02_als_first_run.ipynb` (count-transform ablation),
baseline row in `outputs/experiments/log.jsonl`, `tests/test_als_model.py`.
**Exit:** `python -m src.harness.run_search` logs NDCG@10=0.138+/-0.002 (3 seeds);
12 tests pass. **Done 2026-06-30.**
**Resolved:** `implicit` installs and runs on Python 3.14 in the `.venv`; no
fallback interpreter needed. BLAS pinned to 1 thread (kernel env + run_search).

## Milestone 4 — Autoresearch search session + cold-start fallback (complete)
**Goal:** Run the full disciplined search defined in `program.md` — the bounded,
pre-registered propose -> evaluate -> log -> rank session (the karpathy
autoresearch loop, with auto-promotion inverted to a human gate). Implement
`cold_start.py`. Select a best config by hand with reasoning.
**Artifacts:** `run_session.py` (12 pre-registered configs), filled `log.jsonl`,
`confirm_holdout.py`, `cold_start.py` + tests, `decisions.md` entry naming the
chosen config (factors=32, reg=0.01, iters=15, alpha=1.0),
`notebooks/03_search_readout.ipynb`, `docs/specs/autoresearch_mapping.md`.
**Exit:** Chosen config CV NDCG@10=0.171; locked holdout read once -> 0.233; 16
tests pass. **Done 2026-06-30.**

## Milestone 5 — Model card & write-up
**Goal:** Document how recommendations are generated and how cold start is
handled.
**Artifacts:** `docs/specs/model_card.md`, `outputs/reports/` summary.

## Milestone 6 — FastAPI service (complete)
**Goal:** `GET /recommendations/{user_id}` serving the frozen config, with the
cold-start path wired in.
**Artifacts:** working `api/main.py` (lifespan-loaded model, pydantic response,
ALS + cold-start strategies), `tests/test_api.py` (4 passing), README usage +
example responses.
**Exit:** Verified live with uvicorn — known user 2 returns ALS recs (Pet Shop
Boys, U2, a-ha, ...), unknown user returns popularity (Lady Gaga, Britney, ...),
k bounds enforced (422). **Done 2026-06-30.**

## Phase 2 — Scale-up & model exploration (complete)
**Goal:** Test whether the Phase-1 conclusions survive real data, and whether a
more powerful model can beat the chosen one.
**Artifacts:** `src/data_360k.py` (Last.fm-360K core), rebuilt split,
`notebooks/07` (deep-vs-simple on 2k), `notebooks/08` (360K scale-up),
`src/exp_deep_360k.py` (deep-learning capstone), `src/serving.py` (shared core),
`app/` (Streamlit), `report/` (Quarto), refreshed `model_card.md` + `decisions.md`.
**Exit:** On 360K the ranking flipped — **EASE** (linear autoencoder) is served,
significantly beating ALS. A deep **Mult-VAE** gauntlet confirmed capacity pays off
with data (it climbs from last to 2nd) but still loses to EASE on accuracy, while
reaching ~2x the catalogue — the accuracy-vs-discovery frontier. 53 tests pass.
**Done 2026-07-01.**

## If we slip
The milestones are a rhythm, not a deadline. Partial completion still ships
value — Milestone 1 + 2 alone (a trustworthy harness on a well-understood
dataset) is a genuine artifact. Note slips in `decisions.md`.
