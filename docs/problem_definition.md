# Problem Definition — Scope Contract

## Goal
Build a music recommender on the Last.fm HetRec 2011 user–artist dataset, tune
its collaborative-filtering hyperparameters with a disciplined, auditable search
against a frozen evaluation harness, and serve recommendations over a FastAPI
JSON endpoint.

## Why this project
Portfolio piece for Spotify DS applications — deliberately music-domain. It also
demonstrates a reusable, auditable ML-search discipline (the three-file
architecture) carried over from a professional FPD project.

## In scope
- EDA quantifying sparsity and the per-user interaction distribution.
- A frozen evaluation layer: per-user interaction holdout split + precision@k /
  recall@k / NDCG@k, toy-validated.
- Implicit-feedback ALS (listening counts), tuned via the three-file search.
- A cold-start fallback (popularity / content) for users ALS can't serve.
- A FastAPI `GET /recommendations/{user_id}` endpoint returning JSON.
- Confluence-style documentation of the architecture and decisions.

## Out of scope
- Any frontend / Streamlit (a JSON endpoint is the deliverable).
- Deep-learning recommenders (two-tower, sequence models) — possible "future
  work" note, not built.
- Social-graph (`user_friends.dat`) and tag-based signals as primary model
  inputs — tags are reserved for the optional content fallback only.
- Online / real-time learning. The model is fit offline on a fixed snapshot.

## Feedback framing
**Implicit.** The signal is a listening count per (user, artist), not an explicit
rating. Modeled with confidence-weighted ALS; evaluated with ranking metrics.
(decisions.md, 2026-06-30.)

## Success criteria
- A trustworthy, toy-validated evaluation harness.
- A search log (`outputs/experiments/log.jsonl`) of pre-registered configs with
  metrics, and a human-selected best config recorded in `decisions.md` with
  reasoning.
- A working, documented API endpoint with a sane cold-start path.
- Documentation a reviewer can read to understand *how* recommendations are
  generated and *how* cold start is handled.

## Non-goals masquerading as goals (explicitly rejected)
- Chasing a single high metric number. At this sparsity, a wide enough search
  manufactures lucky winners; the protocol (`program.md`) guards against that.
- Porting FPD's stability-across-seeds machinery by default — only if the
  sparsity investigation justifies it.
