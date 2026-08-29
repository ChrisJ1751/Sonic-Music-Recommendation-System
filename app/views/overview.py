"""Overview — the dashboard landing.

At-a-glance state of the project: how good the model is, what it beat, and where
to go next. The *narrative* lives in the Quarto report; this app is the
play-with-it surface, so this page leads with numbers and routes, and keeps prose
in an expander rather than an essay above the fold.
"""
from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st
from _shared import FAINT, GREEN, MUTED, card, data_stats, get_about, kpi_row, page_header, style_fig

# Default to the PUBLISHED surfaces, override for local work. The previous
# default was http://localhost:8080, which meant the deployed app shipped a dead
# link to every visitor whenever REPORT_URL was not set on the Space -- which it
# was not. Defaulting to production makes the common case correct by construction.
REPORT_URL = os.environ.get("REPORT_URL", "https://chrisj1751.github.io/Sonic-Music-Recommendation-System/")
API_URL = os.environ.get("API_URL", "https://jone1751-sonic-api.hf.space")

about = get_about()
d = data_stats()

page_header("Sonic", "Music recommendations from listening patterns · Last.fm-360K · served model: EASE")

# ---- band 1: how good is it -------------------------------------------------
kpi_row([(m["label"], f"{m['value']:.3f}", m["note"]) for m in about["headline"]])
st.caption(f"Full-catalogue ranking over all {d['n_items']:,} artists — no sampled-negative shortcuts. "
           "Every model ranked the same frozen split.")

# Dataset shape as one compact line rather than a second tile band. Tiles stack
# on narrow viewports, and eight full-width number cards pushed everything below
# the fold on a phone -- these numbers also get full tiles on the Data page.
st.caption(
    f"Trained on **{d['n_users']:,} users** × **{d['n_items']:,} artists** "
    f"({d['nnz']:,} interactions, {d['density'] * 100:.2f}% dense)."
)

st.write("")

# ---- main grid: the result, and the model that produced it ------------------
left, right = st.columns([1.35, 1], gap="medium")

with left:
    with card("Model comparison", "NDCG@10 on the held-out 360K split. Green is the served model."):
        lb = about["leaderboard"][::-1]
        fig = go.Figure(go.Bar(
            x=[r["ndcg10"] for r in lb], y=[r["model"] for r in lb], orientation="h",
            marker_color=[GREEN if r["served"] else FAINT for r in lb],
            text=[f"{r['ndcg10']:.3f}" for r in lb], textposition="outside", cliponaxis=False,
        ))
        fig.update_layout(xaxis_range=[0, 0.26], showlegend=False)
        st.plotly_chart(style_fig(fig, 260), width="stretch")

    with card("Is the gap real?"):
        st.success(about["significance"])

with right:
    with card("Served model — EASE", about["model"]["long"]):
        st.caption(about["model"]["kind"])
        st.code(about["model"]["detail"], language="text")

    with card("Why not deep learning?"):
        st.info(about["pivot"])

st.write("")

# ---- routes: where to go next -----------------------------------------------
st.markdown("##### Explore")
routes = [
    ("views/recommendations.py", "🎧", "Recommendations", "Pick a listener, see what EASE suggests next."),
    ("views/artist_radio.py", "📻", "Artist radio", "Artist-to-artist neighbours from co-listening."),
    ("views/data.py", "📊", "The data", "Live EDA of the 360K core and the long tail."),
    ("views/results.py", "🏆", "Models & results", "Leaderboard, significance, coverage and novelty."),
]
for col, (target, icon, label, blurb) in zip(st.columns(4), routes, strict=True):
    with col, st.container(border=True):
        st.page_link(target, label=f"**{label}**", icon=icon)
        st.caption(blurb)

# ---- the narrative, deliberately below the fold and collapsed ---------------
with st.expander("About this project — the one-paragraph version"):
    st.markdown(
        "A collaborative-filtering music recommender, built and evaluated the way a research team would "
        "ship one — **frozen metrics, a leakage-safe holdout, strong baselines, and significance tests**. "
        "The served model is **EASE**, a linear item-item autoencoder that beat tuned ALS *and* a deep "
        "Mult-VAE on real, uncapped listening data.\n\n"
        "The deliverable is not *“I trained EASE.”* It is *“I built an evaluation process trustworthy "
        "enough that it told me to change my mind.”* **Methodology & limitations** has the discipline; "
        "**The pivot** section of the report has the story of the reversal."
    )

st.caption(
    f"📄 [Read the written report]({REPORT_URL}) · "
    f"⚙️ [API explorer]({API_URL}/docs) · "
    f"[`/health`]({API_URL}/health) — the same recommendations over HTTP, from a containerised service."
)
st.markdown(
    f"<div style='color:{MUTED};font-size:12px;margin-top:4px'>"
    "Report = read-the-work · App = play-with-it · API = use-it-from-code</div>",
    unsafe_allow_html=True,
)
