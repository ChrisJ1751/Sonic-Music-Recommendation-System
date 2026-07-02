"""Methodology & limitations — the discipline that makes the numbers trustworthy,
how a recommendation is actually generated, and an honest account of the limits."""
from __future__ import annotations

import streamlit as st
from _shared import get_about, page_header

about = get_about()

page_header("Methodology & limitations",
            "Why the numbers are trustworthy — and an honest account of where the model falls short.")

# ---- how a recommendation is made ----
st.markdown("#### How a recommendation is generated")
steps = [
    ("1. Look up", "Find the user's row in the binarised interaction matrix (who they've listened to)."),
    ("2. Score", "EASE scores every one of the 11,607 artists in a single sparse mat-vec: `scores = x · B`, "
                 "where `B` is the learned item-item weight matrix."),
    ("3. Mask", "Zero out artists the user has already played, so recommendations are genuinely new."),
    ("4. Rank / diversify", "Take the top-k. If `diversity > 0`, re-rank a wider pool with MMR (using ALS "
                            "item embeddings) to trade a little accuracy for a more varied list."),
]
cols = st.columns(4)
for col, (t, b) in zip(cols, steps, strict=True):
    with col:
        st.markdown(f"**{t}**")
        st.caption(b)
st.caption("Unknown users (past the end of the matrix) fall back to a popularity recommender — the safe "
           "cold-start path.")

st.divider()

# ---- methodology cards ----
st.markdown("#### The discipline behind the numbers")
st.caption("Every reported figure is auditable; nothing was tuned against the sealed holdout.")
mcards = about["methodology"]
for start in range(0, len(mcards), 3):
    row = st.columns(3, gap="medium")
    for col, card in zip(row, mcards[start:start + 3], strict=False):
        with col:
            st.markdown(f"**{card['title']}**")
            st.caption(card["body"])

st.divider()

# ---- limitations (honest) ----
st.markdown("#### Limitations & known trade-offs")
lims = [
    ("Long-tail items are near-unrecommendable",
     "Collaborative filtering has almost nothing to learn for artists with very few listeners. This caps "
     "achievable recall and is a property of the data, not a bug."),
    ("Popularity bias / feedback loops",
     "Like all CF, the model leans toward already-popular artists; if its outputs fed back into training it "
     "would reinforce that. Not mitigated here (offline, single snapshot)."),
    ("No temporal signal",
     "The snapshot has no usable timestamps, so the model cannot capture trend or recency."),
    ("Stale data",
     "Last.fm-360K is a research snapshot of past listening — not representative of today's catalogue or taste."),
    ("No content features",
     "This is pure collaborative filtering; a production system would blend in audio/tag features to help "
     "genuine cold-start artists the CF model can't reach."),
    ("Coverage vs accuracy is a product call",
     "The served config favours accuracy over catalogue coverage. The `diversity` lever exposes the trade-off "
     "rather than hard-coding a single answer."),
]
for start in range(0, len(lims), 2):
    row = st.columns(2, gap="large")
    for col, (t, b) in zip(row, lims[start:start + 2], strict=False):
        with col:
            st.markdown(f"**{t}**")
            st.caption(b)

st.divider()
st.markdown("#### Reproducibility")
st.caption("Deterministic given seeds (BLAS pinned to one thread). 53 tests, ruff-clean, CI on every push. "
           "The full pipeline rebuilds from `python -m src.data_360k` → `make_split` → `run_session` → "
           "`confirm_holdout`; notebooks 00–08 reproduce the EDA, methodology, and the 2k → 360K scale-up.")
tags = "  ".join(f"`{t}`" for t in about["stack"])
st.markdown(f"**Stack**  {tags}")
