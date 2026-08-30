"""Methodology & limitations, the discipline that makes the numbers trustworthy,
how a recommendation is actually generated, and an honest account of the limits.

Card-based rather than prose: each claim is a panel you can scan, with the
longer argument behind an expander. The report is the read-it-through version.
"""
from __future__ import annotations

import streamlit as st
from _shared import card, get_about, page_header

about = get_about()

page_header("Methodology & limitations",
            "Why the numbers are trustworthy, and an honest account of where the model falls short.")

# ---- the serving path, as a pipeline ----------------------------------------
st.markdown("##### How a recommendation is generated")
steps = [
    ("1 · Look up", "Find the user's row in the binarised interaction matrix, who they have listened to."),
    ("2 · Score", "EASE scores all 11,607 artists in one sparse mat-vec: `scores = x · B`, where `B` is the "
                  "learned item-item weight matrix."),
    ("3 · Mask", "Zero out artists the user has already played, so every recommendation is genuinely new."),
    ("4 · Rank", "Take the top-k. With `diversity > 0`, re-rank a wider pool with MMR to trade a little "
                 "accuracy for a more varied list."),
]
for col, (title, body) in zip(st.columns(4), steps, strict=True):
    with col, st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(body)
st.caption("Unknown users, past the end of the matrix, fall back to a popularity recommender. That is the "
           "safe cold-start path, and the API reports it as `cold_start_popularity` rather than hiding it.")

st.write("")

# ---- the discipline, straight from the served payload -----------------------
st.markdown("##### The discipline behind the numbers")
st.caption("Every reported figure is auditable; nothing was tuned against the sealed holdout.")
entries = about["methodology"]
for start in range(0, len(entries), 3):
    row = st.columns(3, gap="medium")
    for col, entry in zip(row, entries[start:start + 3], strict=False):
        with col, st.container(border=True):
            st.markdown(f"**{entry['title']}**")
            st.caption(entry["body"])

st.write("")

# ---- limitations, stated plainly --------------------------------------------
st.markdown("##### Limitations & known trade-offs")
limits = [
    ("Long-tail items are near-unrecommendable",
     "Collaborative filtering has almost nothing to learn for artists with very few listeners. This caps "
     "achievable recall and is a property of the data, not a bug."),
    ("Popularity bias / feedback loops",
     "Like all CF, the model leans toward already-popular artists; if its outputs fed back into training it "
     "would reinforce that. Not mitigated here, offline, single snapshot."),
    ("No temporal signal",
     "The snapshot has no usable timestamps, so the model cannot capture trend or recency."),
    ("Stale data",
     "Last.fm-360K is a research snapshot of past listening, not today's catalogue or taste."),
    ("No content features",
     "Pure collaborative filtering. A production system would blend in audio/tag features to reach genuine "
     "cold-start artists that CF cannot."),
    ("Coverage vs accuracy is a product call",
     "The served config favours accuracy over catalogue coverage. The `diversity` lever exposes the "
     "trade-off rather than hard-coding one answer."),
]
for start in range(0, len(limits), 3):
    row = st.columns(3, gap="medium")
    for col, (title, body) in zip(row, limits[start:start + 3], strict=False):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(body)

st.write("")

# ---- reproducibility --------------------------------------------------------
repro, stack = st.columns([1.4, 1], gap="medium")

with repro:
    with card("Reproducibility", "Deterministic given seeds; BLAS pinned to one thread."):
        st.markdown(
            "The full pipeline rebuilds from `python -m src.data_360k` → `make_split` → `run_session` → "
            "`confirm_holdout`. Notebooks `00`-`08` reproduce the EDA, the harness validation, the "
            "disciplined search, and the 2k → 360K scale-up."
        )
        st.caption("137 tests, ruff-clean, CI on every push. Including a container build, so the deployed "
                   "image cannot drift from the tested code.")

with stack:
    with card("Stack"):
        st.markdown("  ".join(f"`{tool}`" for tool in about["stack"]))
        st.caption("One shared inference core (`src/serving.py`) behind three surfaces: this app, the "
                   "FastAPI service, and the Quarto report.")
