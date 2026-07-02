# Music Recommender — Project Home (Confluence-style index)

A reader-facing landing page. Start here, then follow the links. The dense
technical log lives in [`decisions.md`](../../decisions.md); these pages are the
narrative.

## At a glance
| | |
|---|---|
| **What** | Collaborative-filtering music recommender (Last.fm-360K served; Last.fm-2k for Phase-1 methodology) |
| **Why** | Portfolio project for Spotify DS roles; demonstrates disciplined, auditable ML search |
| **Signal** | Implicit feedback (per-user-artist listening counts) |
| **Deliverable** | Three surfaces from one shared core: a **Streamlit** app, a **FastAPI** service, and a **Quarto** report |
| **Result** | **EASE** served: NDCG@10 0.219 on 360K (NDCG@100 0.36 / Recall@50 0.42, SOTA band); beats a deep Mult-VAE and tuned ALS, all p<0.001 |
| **Status** | Phase 1 + Phase 2 complete (see [phase plan](../phase_plan.md)); next: track-level recs on a newer dataset |

## The pages
- **[Problem definition](../problem_definition.md)** — scope contract: what's in,
  what's out, success criteria.
- **[Milestone plan](../phase_plan.md)** — the six milestones and their artifacts.
- **[Evaluation design](../specs/eval_design.md)** — *why* the train/test split
  is per-user (and why naive splits leak), and what the metrics mean.
- **[Sparsity & fragility investigation](../specs/sparsity_fragility_investigation.md)**
  — the honest call on whether this dataset needs the FPD-style stability layer.
- **[Model card](../specs/model_card.md)** — how recommendations are generated,
  cold-start handling, metrics, limitations, fairness.
- **[Results summary](../../outputs/reports/results_summary.md)** — the one-page
  outcome for a non-technical reader.
- **Written report** — [`report/`](../../report): a Quarto website (data →
  methodology → results → **model exploration** → the pivot → limitations). Render
  with `quarto render report`.
- **Interactive app** — [`app/`](../../app): a Streamlit front door (live
  recommender, artist radio, results and charts). Run `streamlit run
  app/streamlit_app.py`.
- **[Autoresearch mapping](../specs/autoresearch_mapping.md)** — how the search
  harness adapts (and deliberately inverts) karpathy/autoresearch.
- **[Decision log](../../decisions.md)** — every settled choice, newest first.
- **The search architecture** — [`src/harness/`](../../src/harness): the frozen
  `eval_core.py`, the editable `search_config.py`, the `program.md` protocol.
- **Notebooks** — [`00`](../../notebooks/00_eda_interaction_matrix.ipynb) EDA,
  [`01`](../../notebooks/01_evaluation_harness.ipynb) harness validation,
  [`02`](../../notebooks/02_als_first_run.ipynb) ALS ablation,
  [`03`](../../notebooks/03_search_readout.ipynb) search readout,
  [`04`](../../notebooks/04_recommendation_explorer.ipynb) qualitative explorer
  (item similarity, archetypes, popularity bias),
  [`05`](../../notebooks/05_model_comparison.ipynb) model comparison vs strong
  baselines (BPR, item-item BM25, popularity),
  [`06`](../../notebooks/06_content_cold_start.ipynb) content-based similarity for
  the long tail,
  [`07`](../../notebooks/07_deep_vs_simple.ipynb) deep-vs-simple on 2k,
  [`08`](../../notebooks/08_scaling_to_360k.ipynb) scaling to Last.fm-360K.

## The one idea to take away
The thing that *defines a win* (the split + metrics, in `eval_core.py`) is
**frozen** and can never be edited by the thing *trying to win* (the search loop,
which may only touch `search_config.py`). That separation is what makes the
hyperparameter search auditable rather than a story told after the fact.
