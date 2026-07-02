"""Models & results — the leaderboard, significance, calibration against SOTA,
live beyond-accuracy metrics, and the deep-vs-simple story behind the pivot."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from _shared import (
    FAINT,
    FIG_DIR,
    GREEN,
    PURPLE,
    get_about,
    page_header,
    style_fig,
)

about = get_about()

page_header("Models & results", "How the served model was chosen — measured honestly, on a frozen harness.")

# ---- leaderboard ----
st.markdown("#### The leaderboard")
st.caption("NDCG@10 on the held-out Last.fm-360K split, full-catalogue ranking. Every contender ran through "
           "the same frozen split and metrics — EASE was chosen because it won, not by preference.")
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

st.divider()

# ---- calibration vs SOTA ----
lc, rc = st.columns([1.15, 1], gap="large")
with lc:
    st.markdown("#### Is 0.22 low? The cutoff curve says no")
    st.caption("NDCG@10 reads low only because we rank the full catalogue at a tight cutoff. The dotted lines "
               "mark published SOTA at *its* cutoff (Recall@50, NDCG@100, ...); our curve reaches them there — "
               "the tight @10 view is just the least flattering slice of a strong ranker.")
    cc = about["cutoff_curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cc["k"], y=cc["ndcg"], mode="lines+markers",
                             name="our NDCG@k", line=dict(color=GREEN, width=3)))
    fig.add_trace(go.Scatter(x=cc["k"], y=cc["recall"], mode="lines+markers",
                             name="our Recall@k", line=dict(color=PURPLE, width=3)))
    # SOTA references as dotted lines coloured to match their metric, labelled with the cutoff.
    for p in cc["sota_points"]:
        color = GREEN if p["metric"] == "ndcg" else PURPLE
        fig.add_hline(y=p["value"], line_dash="dot", line_color=color, opacity=0.6,
                      annotation_text=f"SOTA {p['metric'].upper()}@{p['k']} = {p['value']:.2f}",
                      annotation_position="right",
                      annotation_font=dict(size=9, color=color))
    fig.update_layout(title="EASE across cutoffs vs published SOTA (dotted = SOTA at its reported cutoff)",
                      xaxis_title="cutoff k", yaxis_title="metric value", yaxis_range=[0.15, 0.57])
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)
with rc:
    st.markdown("#### Reported metrics")
    st.caption("Full-ranking, macro-averaged over scored users.")
    for m in about["headline"]:
        st.metric(m["label"], f"{m['value']:.3f}", help=m["note"])

st.divider()

# ---- beyond accuracy: the accuracy-vs-coverage frontier ----
st.markdown("#### Beyond accuracy — the deep model reaches widest")
st.caption("Accuracy is not the whole story. On the full held-out set we also measure catalog **coverage** "
           "(fraction of the 11,607 artists ever recommended) and **novelty**. The deep Mult-VAE covers far "
           "more of the catalogue than EASE — but at lower top-10 accuracy. That is the accuracy-vs-discovery "
           "trade-off, not a free lunch.")
ba = about["beyond_accuracy"]
b1, b2 = st.columns([1, 1], gap="large")
with b1:
    badf = pd.DataFrame(ba)[["model", "ndcg10", "coverage", "novelty"]]
    st.dataframe(
        badf, hide_index=True, use_container_width=True,
        column_config={
            "model": st.column_config.TextColumn("model"),
            "ndcg10": st.column_config.NumberColumn("NDCG@10", format="%.3f"),
            "coverage": st.column_config.NumberColumn("coverage", format="%.3f",
                                                      help="fraction of all 11,607 artists ever recommended"),
            "novelty": st.column_config.NumberColumn("novelty (bits)", format="%.2f",
                                                     help="mean self-information of recommended artists"),
        },
    )
with b2:
    fig = go.Figure()
    for r in ba:
        color = GREEN if r["served"] else (PURPLE if "VAE" in r["model"] else FAINT)
        fig.add_trace(go.Scatter(
            x=[r["coverage"]], y=[r["ndcg10"]], mode="markers+text",
            text=[r["model"]], textposition="top center", showlegend=False,
            textfont=dict(size=10, color=color),
            marker=dict(size=17 if r["served"] else 11, color=color,
                        line=dict(color="#ffffff", width=1.2 if r["served"] else 0))))
    fig.update_layout(title="Accuracy vs coverage on 360K",
                      xaxis_title="catalog coverage", yaxis_title="NDCG@10",
                      xaxis_range=[-0.03, 0.92], yaxis_range=[0.02, 0.25])
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)
st.info("EASE ranks best; Mult-VAE reaches ~2x the catalogue (0.81 vs 0.42) at lower NDCG@10. The "
        "`diversity` slider on the **Recommendations** page is the runtime lever on the same trade-off.")

st.divider()

# ---- the pivot story ----
st.markdown("#### Why the served model is EASE, not deep learning")
st.caption(about["pivot"])
p1, p2 = st.columns(2)
for col, fname, cap in [
    (p1, "deep_vs_simple.png", "Phase 1 (2k): a deep VAE and linear EASE vs tuned ALS — ALS holds on small data."),
    (p2, "ranking_flip.png", "Capacity pays off on real data: on 360K both EASE and the deep VAE overtake ALS."),
]:
    p = FIG_DIR / fname
    if p.exists():
        col.image(str(p), caption=cap, use_container_width=True)
