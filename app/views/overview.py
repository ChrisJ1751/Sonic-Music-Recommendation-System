"""Overview — the 30-second pitch: what the project is, the headline result, and
where to go for the full story. Deeper pages live in the sidebar."""
from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st
from _shared import FAINT, GREEN, data_stats, get_about, page_header, style_fig

# Set REPORT_URL in the deployment (e.g. the GitHub Pages URL) to link the report.
REPORT_URL = os.environ.get("REPORT_URL", "http://localhost:8080")

about = get_about()
d = data_stats()

page_header("Sonic", "Music recommendations from listening patterns · Last.fm-360K · served model: EASE")

st.markdown(
    "A collaborative-filtering music recommender, built and evaluated the way a research team would ship one "
    "— **frozen metrics, a leakage-safe holdout, strong baselines, and significance tests**. The served model "
    "is **EASE**, a linear item-item autoencoder that beat tuned ALS *and* a deep VAE on real, uncapped "
    "listening data. This app is the whole story: the data, the model comparison, the methodology, and a live demo."
)

# ---- headline metrics ----
st.subheader("Results at a glance")
st.caption(f"Full-catalogue ranking over all {d['n_items']:,} artists — no sampled-negative shortcuts.")
cols = st.columns(len(about["headline"]))
for col, m in zip(cols, about["headline"], strict=True):
    col.metric(m["label"], f"{m['value']:.3f}", help=m["note"])
    col.caption(m["note"])

st.divider()
left, right = st.columns(2, gap="large")

with left:
    st.markdown("#### How it stacks up")
    st.caption("NDCG@10 on the held-out 360K split. EASE is served.")
    lb = about["leaderboard"][::-1]
    colors = [GREEN if r["served"] else FAINT for r in lb]
    fig = go.Figure(go.Bar(
        x=[r["ndcg10"] for r in lb], y=[r["model"] for r in lb], orientation="h",
        marker_color=colors, text=[f"{r['ndcg10']:.3f}" for r in lb],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(title="Model comparison (NDCG@10)", xaxis_range=[0, 0.26], showlegend=False)
    st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    st.success(about["significance"])

with right:
    st.markdown("#### The model — EASE")
    st.caption(about["model"]["kind"] + " · " + about["model"]["long"])
    st.markdown(f"`{about['model']['detail']}`")
    st.markdown("#### Why EASE, not deep learning?")
    st.info(about["pivot"])

st.divider()

# ---- dataset one-liner + routing ----
st.subheader("Explore the whole story")
r = st.columns(3)
r[0].metric("Users", f"{d['n_users']:,}")
r[1].metric("Artists", f"{d['n_items']:,}")
r[2].metric("Interactions", f"{d['nnz']:,}")
st.markdown(
    "- **The data** — live EDA of the 360K core: history lengths, the long tail, concentration, and the "
    "2k contrast that drove the pivot.\n"
    "- **Models & results** — the full leaderboard, significance, SOTA calibration, and *live* "
    "beyond-accuracy metrics (coverage / novelty).\n"
    "- **Methodology & limitations** — how a recommendation is made, the evaluation discipline, and an "
    "honest account of the limits.\n"
    "- **Try it live** — pick a listener for real recommendations, or explore artist-to-artist radio."
)
st.caption("Prefer to read it as a report? The written companion (data → methodology → results → "
           f"model exploration → limitations) is the Quarto site: [open the report]({REPORT_URL}).")
