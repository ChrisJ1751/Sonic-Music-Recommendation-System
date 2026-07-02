"""The data — live EDA over the served Last.fm-360K matrix, plus the 2k contrast
that motivated the pivot. Charts are computed live, so they always match the data
the model is actually trained on."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from _shared import FIG_DIR, GREEN, MUTED, PURPLE, TEXT, data_stats, page_header, style_fig

d = data_stats()

page_header("The data", "Last.fm-360K — real, uncapped listening histories. Everything below is computed live.")

# --- KPI row ---
c = st.columns(4)
c[0].metric("Users", f"{d['n_users']:,}")
c[1].metric("Artists", f"{d['n_items']:,}")
c[2].metric("Interactions", f"{d['nnz']:,}")
c[3].metric("Density", f"{d['density']*100:.2f}%", help="Fraction of the user×artist matrix that is non-zero.")

st.caption(
    f"Each user brings a median of **{d['median_history']} artists** ({d['mean_history']:.0f} on average) — "
    f"real histories, not the 50-artist cap of the small 2k set. The matrix is "
    f"**{(1-d['density'])*100:.1f}% sparse**, the normal regime for collaborative filtering."
)

st.divider()

left, right = st.columns(2, gap="large")

with left:
    st.markdown("#### Artists per user")
    st.caption("How much history each listener brings. Uncapped, unlike the 2k set.")
    upu = d["artists_per_user"]
    upu_plot = upu[upu <= np.percentile(upu, 99)]  # trim the extreme tail for a readable axis
    fig = go.Figure(go.Histogram(x=upu_plot, nbinsx=40, marker_color=GREEN))
    fig.add_vline(x=d["median_history"], line_dash="dash", line_color=TEXT,
                  annotation_text=f"median {d['median_history']}", annotation_font_size=11)
    fig.update_layout(title="Distribution of history length",
                      xaxis_title="artists in a user's history", yaxis_title="users")
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)

with right:
    st.markdown("#### Listeners per artist")
    st.caption("The item side. A healthy core — the 2k set's median artist had a single listener.")
    ipi = d["listeners_per_artist"]
    fig = go.Figure(go.Histogram(x=np.log10(ipi), nbinsx=40, marker_color=PURPLE))
    fig.update_layout(title="Distribution of artist popularity (log scale)",
                      xaxis_title="log10(distinct listeners)", yaxis_title="artists")
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)

st.divider()

lc, rc = st.columns([1.2, 1], gap="large")
with lc:
    st.markdown("#### Listening is highly concentrated")
    st.caption("A Lorenz curve of artist popularity — why a popularity baseline is hard to beat, "
               "and why catalog coverage is a real product concern.")
    ipi = d["listeners_per_artist"]
    order = np.sort(ipi)
    cum = np.cumsum(order) / order.sum()
    xs = np.arange(1, len(order) + 1) / len(order)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=cum, mode="lines", line=dict(color=GREEN, width=3),
                             fill="tonexty", fillcolor="rgba(30,215,96,0.10)", name="artists"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=MUTED, dash="dash", width=1), name="perfect equality"))
    fig.update_layout(title=f"Lorenz curve (Gini {d['gini_listeners']:.2f})",
                      xaxis_title="cumulative share of artists (least → most popular)",
                      yaxis_title="cumulative share of listener-relations")
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)

with rc:
    st.markdown("#### What that means for modelling")
    st.metric("Popularity Gini", f"{d['gini_listeners']:.2f}", help="0 = uniform, 1 = one artist gets everything.")
    st.metric("Top 1% of artists", f"{d['top1pct_share']*100:.0f}% of all listens",
              help="Share of all listener-relations captured by the most popular 1% of artists.")
    st.metric("Play counts", f"1 → {d['max_plays']:,}",
              help=f"median {d['median_plays']}, 99th pct {d['p99_plays']:,.0f} — heavy-tailed, "
                   f"which is why we model implicit confidence, not raw counts (no RMSE).")
    st.caption("Concentration this steep is why the honest baseline to beat is popularity, and why we "
               "report coverage/novelty alongside accuracy (see **Models & results**).")

st.divider()

st.markdown("#### For contrast — the 2k set we started on")
st.caption("The small HetRec-2k set hard-capped every user at 50 artists and left ~61% of artists with a "
           "single listener. That artificial shape is exactly why we pivoted to 360K — and why the model "
           "ranking changed once real data arrived (see **Models & results**).")
g1, g2 = st.columns(2)
for col, fname, cap in [
    (g1, "artist_long_tail.png", "Last.fm-2k: most artists had a single listener — near-unrecommendable."),
    (g2, "lorenz_popularity.png", "Last.fm-2k: an even steeper Lorenz curve (Gini 0.73)."),
]:
    p = FIG_DIR / fname
    if p.exists():
        col.image(str(p), caption=cap, use_container_width=True)
