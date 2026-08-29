"""Models & results — leaderboard, significance, calibration against SOTA, live
beyond-accuracy metrics, and the deep-vs-simple story behind the pivot.

Laid out as a dashboard: numbers and charts lead, the longer arguments sit in
expanders. The full prose version of this page is the Quarto report.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from _shared import (
    FAINT,
    FIG_DIR,
    GREEN,
    LABEL,
    PURPLE,
    card,
    get_about,
    kpi_row,
    page_header,
    style_fig,
)

about = get_about()

page_header("Models & results", "How the served model was chosen — measured honestly, on a frozen harness.")

# ---- band: the reported numbers ---------------------------------------------
kpi_row([(m["label"], f"{m['value']:.3f}", m["note"]) for m in about["headline"]])
st.caption("Full-ranking, macro-averaged over scored users. Every contender ran through the same frozen "
           "split and the same frozen metrics — EASE was chosen because it won, not by preference.")

st.write("")

# ---- grid: leaderboard + significance ---------------------------------------
lead, sig = st.columns([1.3, 1], gap="medium")

with lead:
    with card("The leaderboard", "NDCG@10 on the held-out 360K split, full-catalogue ranking."):
        lb = about["leaderboard"][::-1]
        fig = go.Figure(go.Bar(
            x=[r["ndcg10"] for r in lb], y=[r["model"] for r in lb], orientation="h",
            marker_color=[GREEN if r["served"] else FAINT for r in lb],
            text=[f"{r['ndcg10']:.3f}" for r in lb], textposition="outside", cliponaxis=False,
        ))
        fig.update_layout(xaxis_range=[0, 0.26], showlegend=False)
        st.plotly_chart(style_fig(fig, 280), width="stretch")

with sig:
    with card("Is the gap real?", "Paired user-level bootstrap, 5,000 resamples."):
        st.success(about["significance"])

st.write("")

# ---- grid: calibration + beyond-accuracy table ------------------------------
cal, tbl = st.columns([1.25, 1], gap="medium")

with cal:
    with card("Is 0.22 low? The cutoff curve says no",
              "Dotted lines are published SOTA at *its* reported cutoff."):
        cc = about["cutoff_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cc["k"], y=cc["ndcg"], mode="lines+markers",
                                 name="our NDCG@k", line=dict(color=GREEN, width=3)))
        fig.add_trace(go.Scatter(x=cc["k"], y=cc["recall"], mode="lines+markers",
                                 name="our Recall@k", line=dict(color=PURPLE, width=3)))
        for point in cc["sota_points"]:
            colour = GREEN if point["metric"] == "ndcg" else PURPLE
            # "right" put the label outside the plotting area once this chart
            # moved into a narrower card, clipping it to "S...". Anchor inside.
            fig.add_hline(y=point["value"], line_dash="dot", line_color=colour, opacity=0.6,
                          annotation_text=f"SOTA {point['metric'].upper()}@{point['k']} {point['value']:.2f}",
                          annotation_position="top left",
                          annotation_font=dict(size=9, color=colour))
        fig.update_layout(xaxis_title="cutoff k", yaxis_title="metric value", yaxis_range=[0.15, 0.57])
        st.plotly_chart(style_fig(fig, 300), width="stretch")

        with st.expander("Why the @10 view is the least flattering slice"):
            st.markdown(
                "NDCG@10 reads low only because every model ranks the **full catalogue** at a tight cutoff "
                "— no sampled-negative shortcuts (Krichene & Rendle, KDD 2020), which inflate scores by "
                "ranking against a handful of random negatives instead of all 11,607 artists. Published "
                "SOTA is usually reported at wider cutoffs; our curve meets it there."
            )

with tbl:
    with card("Beyond accuracy", "Coverage and novelty over the full held-out set."):
        badf = pd.DataFrame(about["beyond_accuracy"])[["model", "ndcg10", "coverage", "novelty"]]
        st.dataframe(
            badf, hide_index=True, width="stretch",
            column_config={
                "model": st.column_config.TextColumn("model"),
                "ndcg10": st.column_config.NumberColumn("NDCG@10", format="%.3f"),
                "coverage": st.column_config.NumberColumn("coverage", format="%.3f",
                                                          help="fraction of all 11,607 artists ever recommended"),
                "novelty": st.column_config.NumberColumn("novelty", format="%.2f",
                                                         help="mean self-information (bits) of recommended artists"),
            },
        )

st.write("")

# ---- grid: the accuracy-vs-discovery frontier -------------------------------
front, note = st.columns([1.3, 1], gap="medium")

with front:
    with card("The accuracy–discovery frontier", "Reaching wider costs top-10 accuracy."):
        fig = go.Figure()
        for row in about["beyond_accuracy"]:
            colour = GREEN if row["served"] else (PURPLE if "VAE" in row["model"] else FAINT)
            fig.add_trace(go.Scatter(
                x=[row["coverage"]], y=[row["ndcg10"]], mode="markers+text",
                text=[row["model"]], textposition="top center", showlegend=False,
                # Marker colour encodes the model; the label is text, so it takes
                # the readable token rather than inheriting a 1.90:1 grey.
                textfont=dict(size=10, color=GREEN if row["served"] else LABEL),
                marker=dict(size=17 if row["served"] else 11, color=colour,
                            line=dict(color="#ffffff", width=1.2 if row["served"] else 0))))
        fig.update_layout(xaxis_title="catalog coverage", yaxis_title="NDCG@10",
                          xaxis_range=[-0.03, 0.92], yaxis_range=[0.02, 0.25])
        st.plotly_chart(style_fig(fig, 300), width="stretch")

with note:
    with card("What that means"):
        st.info("EASE ranks best; Mult-VAE reaches ~2x the catalogue (0.81 vs 0.42) at lower NDCG@10.")
        st.caption("The **diversity** slider on the Recommendations page is the runtime lever on this same "
                   "trade-off — MMR re-ranking, exposed by the API as its `diversity` parameter.")
        st.page_link("views/recommendations.py", label="**Try the lever**", icon="🎧")

st.write("")

# ---- the pivot --------------------------------------------------------------
with card("Why the served model is EASE, not deep learning"):
    st.caption(about["pivot"])
    p1, p2 = st.columns(2)
    for col, fname, cap in [
        (p1, "deep_vs_simple.png",
         "Phase 1 (2k): a deep VAE and linear EASE vs tuned ALS — ALS holds on small data."),
        (p2, "ranking_flip.png",
         "Capacity pays off on real data: on 360K both EASE and the deep VAE overtake ALS."),
    ]:
        path = FIG_DIR / fname
        if path.exists():
            col.image(str(path), caption=cap, width="stretch")
