---
title: Sonic Music Recommender
sdk: streamlit
app_file: app/streamlit_app.py
python_version: "3.11"
pinned: false
short_description: A disciplined music recommender — EASE on Last.fm-360K.
---

<!-- The YAML block above is Hugging Face Space config (used when this repo is
     deployed as a Streamlit Space). GitHub renders it as a small table; harmless.
     See DEPLOY.md. -->

# Music Recommendation System — with Disciplined Hyperparameter Search

A collaborative-filtering music recommender built on the **Last.fm (HetRec 2011)**
user–artist listening dataset, with an **autoresearch-style three-file search
architecture** for tuning the model against a *frozen, reusable* evaluation
harness. Presented three ways from one shared core: a **Streamlit** app (interactive
results + live demo), a **FastAPI** serving layer, and a **Quarto** written report.

> Portfolio project (target: Spotify DS roles). The repository and its
> documentation are as much the deliverable as the model is. See
> [`docs/`](docs/) for the Confluence-style project pages.

## Why three files

This project deliberately reuses the discipline from a prior professional
project (First Payment Delinquency / FPD modeling), which adapted
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) into a
three-file shape:

| Role | File | Who edits it |
|---|---|---|
| **Frozen correctness layer** — the train/test split + the metric definitions | [`src/harness/eval_core.py`](src/harness/eval_core.py) | **Never edited** by the search loop |
| **Editable search surface** — ALS hyperparameters only | [`src/harness/search_config.py`](src/harness/search_config.py) | The **only** file the agent/loop mutates |
| **Protocol** — what to try, in what order, what counts as improvement, when to stop | [`src/harness/program.md`](src/harness/program.md) | Human-written; the loop follows it |

The point: the thing that *defines a win* (the split and the metrics) can never
be edited by the thing that's *trying to win*. That's what makes the search
auditable.

**What carries over from FPD, and what does not.** FPD needed a
stability-across-seeds/folds layer because its data was tiny *and severely
imbalanced* (194 positives, 2.43% base rate). This dataset has a different
problem — **extreme sparsity, not class imbalance**. We apply the same
three-file discipline, but we do **not** copy FPD's stability machinery unless
the EDA shows sparsity creates an analogous fragility. That investigation is
tracked in
[`docs/specs/sparsity_fragility_investigation.md`](docs/specs/sparsity_fragility_investigation.md)
— honesty over cargo-culting.

## Data — two phases

The signal throughout is a **listening count** per (user, artist) — **implicit
feedback**, not ratings.

- **Phase 1 — Last.fm-2k** ([GroupLens HetRec 2011](https://grouplens.org/datasets/hetrec-2011/)):
  1,892 users, 17,632 artists, 92,834 records. Used to *develop the methodology*.
  Its top-50-per-user cap made it small and artificial (and depressed scores).
- **Phase 2 — Last.fm-360K** (Celma): 359K users / 17.6M records of *real,
  uncapped* histories, filtered to a dense core (**39,499 users × 11,607 artists,
  1.68M interactions**; `src/data_360k.py`). This is the **active** dataset —
  what the model, API, and demo run on. Switch with `active_dataset` in
  `configs/data_config.yaml`.

Raw files live in `data/raw/` and `data/raw_360k/` (not committed).

## Repo map

```
configs/            data_config.yaml — where the data lives + schema
data/
  raw/              Last.fm .dat files (gitignored)
  interim/          intermediate artifacts
  processed/
    harness/
      _LOCKED_holdout/   the sealed test interactions the search never reads
docs/
  problem_definition.md  scope contract
  phase_plan.md          milestone plan
  specs/                 eval design, sparsity investigation, model card
  learning/              concept notes (why per-user holdout, etc.)
  stakeholder/           Confluence-style narrative pages
notebooks/          00 EDA · 01 harness validation · 02 ALS ablation · 03 search
                    readout · 04 recommendation explorer · 05 model comparison ·
                    06 content cold start · 07 deep vs simple · 08 scaling to 360K
src/
  utils.py          PROJECT_ROOT-anchored paths, load_config, logging, hashing
  plotting.py       reusable, styled charts (notebooks call these; saved to outputs/figures)
  data_loading.py   load Last.fm-2k, build the matrix, dispatch the active dataset
  data_360k.py      load + preprocess Last.fm-360K (the active dataset)
  deep.py           Mult-VAE (deep recommender, the honest DL comparison)
  exp_deep_360k.py  capstone experiment: EASE vs ALS vs Mult-VAE vs baselines on 360K
  report_figures.py regenerates the report/app figures (calibration, ranking-flip, frontier)
  als_model.py      confidence-weighted implicit ALS (log1p) + similarity, recommend
  cold_start.py     popularity (distinct-listener) fallback for unknown users
  content.py        tag TF-IDF content similarity (reaches the cold long tail)
  metrics.py        beyond-accuracy reporting: coverage, novelty, Gini, diversity
  models.py         model zoo + comparison driver (ALS / BPR / item-item BM25 / popularity)
  serving.py        shared inference core (EASE recommend / similar / cold-start); API + app import it
  rerank.py         MMR diversity re-ranking
  stats.py          bootstrap CIs + paired significance tests
  harness/
    eval_core.py     FROZEN: per-user split + precision@k/recall@k/NDCG + scorer
    search_config.py EDITABLE: ALS factors / regularization / iterations / alpha
    program.md       the search protocol
    run_search.py    FROZEN runner: one config -> multi-seed eval -> log.jsonl
    run_session.py   the pre-registered search session (ranked leaderboard)
    make_split.py    one-time human-run split; SOLE owner of the locked-holdout path
    confirm_holdout.py  one-time, human-gated locked-holdout confirmation
api/                FastAPI service: /recommendations (+ diversity), /similar-artists, /about
app/                Streamlit app (interactive front door): views/ + st.navigation
report/             Quarto website (written report): *.qmd + theme.scss + figures/
tests/              toy-validated metrics, ALS, cold-start, stats, serving, API
outputs/
  experiments/      append-only search log (log.jsonl)
  figures/          generated charts
  reports/          exported EDA / results summaries
decisions.md        append-only decision log, newest at top
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# Dataset (already downloaded into data/raw/ during setup):
#   https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip

jupyter lab notebooks/00_eda_interaction_matrix.ipynb
```

## Reproduce the pipeline

```bash
python -m src.harness.make_split        # build the sealed per-user split
python -m src.harness.run_session       # run the pre-registered search -> log.jsonl
python -m src.harness.confirm_holdout   # one-time locked-holdout confirmation
pytest                                  # 45 tests (toy-validated metrics, ALS, cold-start, API)
```

## Development

```bash
pip install -e ".[dev]"   # editable install + dev tools (pyproject is canonical)
ruff check src api tests  # lint
pytest                    # 45 tests
docker build -t music-rec . && docker run -p 8000:8000 music-rec   # containerised API
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest on every push (Python 3.12).

## Running it — three surfaces, one core

The app and the API import the same inference core (`src/serving.py`), so there is
no duplicated recommendation logic; the report is the static written companion.

**1. The Streamlit app** — the interactive front door (results + charts + live demo):

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py        # http://localhost:8501
```

Sidebar — *The project*: **Overview** (headline result + leaderboard),
**The data** (live 360K EDA), **Models & results** (leaderboard, SOTA calibration,
and live coverage/novelty metrics), **Methodology & limitations**; *Try it live*:
**Recommendations** and **Artist radio**.

**2. The FastAPI service** — the serving/engineering layer:

```bash
uvicorn api.main:app --port 8000          # loads the 360K core + EASE at startup
```

Open **/docs** for the Swagger explorer, **/about** for the results payload, or **/**
for the built-in HTML demo.

```bash
$ curl http://127.0.0.1:8000/recommendations/17084?k=5
{"user_id": 17084, "strategy": "ease", "k": 5, "recommendations": [
  {"artist_id": 733, "name": "joy division", "score": 0.51}, ...]}

# a user index past the end of the matrix falls back to popularity:
$ curl http://127.0.0.1:8000/recommendations/99999999?k=5
{"user_id": 99999999, "strategy": "cold_start_popularity", ...}

# trade some accuracy for a more varied list (MMR diversity re-ranking):
$ curl "http://127.0.0.1:8000/recommendations/17084?k=5&diversity=0.7"
{"user_id": 17084, "strategy": "ease+mmr", ...}

# "fans also like" — nearest artists in EASE's item-item weights:
$ curl http://127.0.0.1:8000/similar-artists/733?k=5
{"artist_id": 733, "name": "joy division", "similar": [...]}
```

**3. The written report** (`report/`) — a Quarto website: the read-the-work
companion (data → methodology → results → **model exploration** → the pivot →
limitations), with the figures and tables inline. The *Model exploration* page
walks the full journey from a popularity baseline to EASE to the deep Mult-VAE.
Source is `report/*.qmd`; render with the [Quarto CLI](https://quarto.org):

```bash
quarto render report        # -> report/_site/index.html
quarto preview report       # live-reloading local preview
```

## Where to look for what

- **What this project is / scope:** [`docs/problem_definition.md`](docs/problem_definition.md)
- **The plan:** [`docs/phase_plan.md`](docs/phase_plan.md)
- **What's been decided and why:** [`decisions.md`](decisions.md)
- **How evaluation works (and why naive splits leak):** [`docs/specs/eval_design.md`](docs/specs/eval_design.md)
- **The search protocol:** [`src/harness/program.md`](src/harness/program.md)

## Status

Methodology complete on Phase 1 (2k), then **scaled to Last.fm-360K** (notebook
`08`). On the richer data the **linear EASE autoencoder** (Steck 2019) wins the
model comparison and is the **served** model, scoring **NDCG@10 = 0.22, NDCG@100 =
0.36, Recall@50 = 0.42** under honest full-ranking evaluation — squarely in the
published SOTA band.

A deep-learning capstone (`src/exp_deep_360k.py`) then benchmarked a **Mult-VAE**
against the zoo on the same frozen split. Corrected 360K leaderboard (NDCG@10, all
p<0.001): **EASE 0.219 > Mult-VAE 0.194 > ALS 0.184 > BM25 0.110 > popularity 0.044**.
The deep model climbs from *last* on 2k to *2nd* on 360K — overtaking ALS — but
still trails EASE. It covers ~2x the catalogue (0.81 vs 0.42) at lower accuracy: the
accuracy-vs-discovery frontier, not a free win. EASE stays served.

53 tests pass, ruff clean. Three surfaces from one shared core — a **Streamlit**
app, a **FastAPI** service, and a **Quarto** report (see `report/`, incl. the
*Model exploration* page). Next planned: track-level (song) recommendations, on a
more recent dataset (e.g. the Spotify Million Playlist Dataset). See
[`docs/specs/model_card.md`](docs/specs/model_card.md) and `decisions.md`.
