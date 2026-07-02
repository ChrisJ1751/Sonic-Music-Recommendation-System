# Music Recommender — Project Context for the Agent

Read this file fully at session start. It contains decisions that should not be
relitigated and conventions to follow without prompting.

## What this project is

A collaborative-filtering music recommender, with an autoresearch-style **three-file
disciplined search** against a frozen evaluation harness. Owner: Chris. Purpose:
portfolio project for Spotify DS applications, hence the deliberately music-domain
choice.

**Current state (Phase 2, as of 2026-07-01):** the project scaled from Last.fm-2k to
**Last.fm-360K**, and the served model is now **EASE** (a linear item-item
autoencoder) — it won a significance-tested gauntlet that included a deep Mult-VAE.
It is presented three ways from one shared inference core (`src/serving.py`): a
**Streamlit** app (`app/`), a **FastAPI** service (`api/`), and a **Quarto** report
(`report/`). NOTE: an earlier version of this file said "no Streamlit, no frontend";
the owner explicitly reversed that — the app and report are now core deliverables.

## The three-file architecture — the core discipline

This mirrors the FPD project's adaptation of karpathy/autoresearch. **The file
that defines what a "win" is must never be editable by the loop trying to win.**

1. **`src/harness/eval_core.py` — FROZEN. The agent never edits this.**
   Owns the per-user train/test interaction split and the
   `precision_at_k` / `recall_at_k` / `ndcg_at_k` definitions. Editing it is a
   model-integrity bug, not a tuning choice. It is validated against a hand-built
   toy example (`tests/test_eval_core.py`) so the metrics are trusted before
   they touch real data.

2. **`src/harness/search_config.py` — the ONLY file the agent edits.**
   ALS hyperparameters: `factors`, `regularization`, `iterations`, and the
   implicit-feedback confidence scaling `alpha`. Bounded, documented ranges.
   No executable evaluation logic lives here.

3. **`src/harness/program.md` — human-written; the agent follows it.**
   What to try and in what order, how to log each attempt, what counts as an
   improvement, stopping criteria. The agent does not invent search moves the
   protocol doesn't sanction.

`src/harness/run_search.py` is the **frozen runner**: it imports `eval_core` and
`search_config`, trains, evaluates, and appends one row per attempt to
`outputs/experiments/log.jsonl`. It **promotes nothing automatically** — a human
reads the log and decides.

## Critical facts — do not relitigate

- **Feedback type: IMPLICIT.** The signal is `weight` = play/listen count per
  (user, artist). There are no explicit star ratings. Model accordingly
  (implicit ALS with confidence weighting `c = 1 + alpha * count`), and evaluate
  with ranking metrics (precision@k / recall@k / NDCG), never RMSE on counts.
- **The split is per-user interaction holdout, NOT a naive row split.** A random
  row split leaks and also breaks the task definition for recommenders — see
  [`docs/specs/eval_design.md`](docs/specs/eval_design.md). This is settled.
- **The locked holdout** (`data/processed/harness/_LOCKED_holdout/`) is sealed
  before search and is never read by the loop. Its path is defined **only** in
  `src/harness/make_split.py` (the one-time, human-run split script) — never in
  `eval_core.py` or `run_search.py`, so loop code has no symbol that could load
  it. Reading it is a one-time human-gated step, recorded in `decisions.md`.
- **FPD's stability-across-seeds machinery is NOT assumed here.** FPD needed it
  for tiny+imbalanced data. This dataset's problem is sparsity, not imbalance.
  Whether sparsity creates an analogous evaluation fragility is an open
  *investigation*, not an assumption — see
  [`docs/specs/sparsity_fragility_investigation.md`](docs/specs/sparsity_fragility_investigation.md).
  Do not port FPD's stability layer wholesale; justify anything you add from the
  data's actual properties.

## Cold start

Users/artists with zero interactions cannot be served by ALS (no latent vector).
A popularity/content-based fallback (`src/cold_start.py`) handles them. The EDA
quantifies how many users have very few interactions, which sizes this concern.

## Conventions

- **`decisions.md`:** append-only, newest at top. Never delete an entry —
  supersede it with a new one that cites the old. Convert relative dates to
  absolute (today is the date in the entry header).
- **Paths go through `src/utils.py`.** It anchors everything to `PROJECT_ROOT`
  (`PROCESSED_DIR`, `RAW_DIR`, `EXPERIMENTS_DIR`, …) and provides `load_config`,
  `get_logger`, `file_sha256`, `make_run_dir`. Never hardcode a relative data
  path or `os.chdir` — import the constant so code works from any cwd.
- **`src/` is reusable code** imported by notebooks and the API. **`notebooks/`
  is for guided analysis**, numbered (`00_…`, `01_…`). Notebooks import from
  `src` (via a one-line `sys.path` insert); they do not redefine pipeline logic
  inline.
- **Frozen files** (`eval_core.py`, the locked holdout, metric defs) are off
  limits to the search loop. If a change to one seems necessary, stop and raise
  it with the human.
- **`outputs/experiments/log.jsonl`** is written only by the runner, never
  hand-edited.
- **Confluence-style docs** live in `docs/` as Markdown. Keep the technical
  decision log (`decisions.md`) and the narrative/stakeholder pages
  (`docs/stakeholder/`) separate — different audiences.

## How Chris works

- Strong analyst, real ML experience but values *why before how* on subtle
  recsys-specific pitfalls (e.g. why naive splitting leaks).
- Pushes back on errors; domain correction usually wins.
- Wants honesty over pattern-matching: if something from FPD doesn't apply here,
  say so rather than copying it.

## Environment note

Local Python is **3.14**. `implicit` and some ML wheels may not yet publish
cp314 builds — confirm install before relying on them in a modeling milestone
(EDA only needs pandas/numpy/scipy/matplotlib/seaborn, which are present).
