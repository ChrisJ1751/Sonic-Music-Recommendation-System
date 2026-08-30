"""The data — live EDA over the served Last.fm-360K matrix, plus the 2k contrast
that motivated the pivot.

Every chart here is computed from the matrix the model is actually serving, so
these panels cannot drift from the model the way an authored figure can.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from _shared import (
    FIG_DIR,
    GREEN,
    MUTED,
    PURPLE,
    TEXT,
    card,
    data_stats,
    kpi_row,
    page_header,
    style_fig,
)

d = data_stats()

page_header("The data", "Last.fm-360K — real, uncapped listening histories. Everything here is computed live.")

# ---- band: shape of the matrix ----------------------------------------------
kpi_row([
    ("Users", f"{d['n_users']:,}", "Listeners in the dense 360K core."),
    ("Artists", f"{d['n_items']:,}", "The full catalogue every model ranks."),
    ("Interactions", f"{d['nnz']:,}", "User–artist pairs with at least one play."),
    ("Density", f"{d['density'] * 100:.2f}%", "Fraction of the user×artist matrix that is non-zero."),
])
st.caption(
    f"Median **{d['median_history']} artists** per user ({d['mean_history']:.0f} on average) — real histories, "
    f"not the 50-artist cap of the small 2k set. **{(1 - d['density']) * 100:.1f}% sparse**, the normal regime "
    "for collaborative filtering."
)

st.write("")

# ---- grid: the two sides of the matrix --------------------------------------
users, items = st.columns(2, gap="medium")

with users:
    with card("Artists per user", "How much history each listener brings. Uncapped, unlike the 2k set."):
        upu = d["artists_per_user"]
        upu_plot = upu[upu <= np.percentile(upu, 99)]   # trim the extreme tail for a readable axis
        fig = go.Figure(go.Histogram(x=upu_plot, nbinsx=40, marker_color=GREEN))
        fig.add_vline(x=d["median_history"], line_dash="dash", line_color=TEXT,
                      annotation_text=f"median {d['median_history']}", annotation_font_size=11)
        fig.update_layout(xaxis_title="artists in a user's history", yaxis_title="users")
        st.plotly_chart(style_fig(fig, 300), width="stretch")

with items:
    with card("Listeners per artist", "The item side. The 2k set's median artist had a single listener."):
        ipi = d["listeners_per_artist"]
        fig = go.Figure(go.Histogram(x=np.log10(ipi), nbinsx=40, marker_color=PURPLE))
        fig.update_layout(xaxis_title="log10(distinct listeners)", yaxis_title="artists")
        st.plotly_chart(style_fig(fig, 300), width="stretch")

st.write("")

# ---- grid: concentration ----------------------------------------------------
lorenz, meaning = st.columns([1.3, 1], gap="medium")

with lorenz:
    with card(f"Listening is highly concentrated (Gini {d['gini_listeners']:.2f})",
              "Why a popularity baseline is hard to beat."):
        ipi = d["listeners_per_artist"]
        order = np.sort(ipi)
        cum = np.cumsum(order) / order.sum()
        xs = np.arange(1, len(order) + 1) / len(order)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=cum, mode="lines", line=dict(color=GREEN, width=3),
                                 fill="tonexty", fillcolor="rgba(30,215,96,0.10)", name="artists"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color=MUTED, dash="dash", width=1), name="perfect equality"))
        fig.update_layout(xaxis_title="cumulative share of artists (least → most popular)",
                          yaxis_title="cumulative share of listener-relations")
        st.plotly_chart(style_fig(fig, 320), width="stretch")

with meaning:
    with card("What that means for modelling"):
        st.metric("Popularity Gini", f"{d['gini_listeners']:.2f}",
                  help="0 = uniform, 1 = one artist gets everything.", border=True)
        st.metric("Top 1% of artists", f"{d['top1pct_share'] * 100:.0f}% of listens",
                  help="Share of all listener-relations captured by the most popular 1% of artists.",
                  border=True)
        st.metric("Play counts", f"1 → {d['max_plays']:,}",
                  help=f"median {d['median_plays']}, 99th pct {d['p99_plays']:,.0f}", border=True)
        with st.expander("Why this shapes the model"):
            st.markdown(
                "Play counts are **heavy-tailed** across six orders of magnitude, which is why the model "
                "uses implicit *confidence* weighting on log-scaled counts rather than treating a count as "
                "a rating — no RMSE anywhere in this project.\n\n"
                "Concentration this steep is also why the honest baseline to beat is **popularity**, not a "
                "random ranker, and why coverage and novelty are reported alongside accuracy."
            )

st.write("")

# ---- the contrast that drove the pivot --------------------------------------
with card("For contrast — the 2k set we started on",
          "Hard-capped at 50 artists per user, ~61% of artists with a single listener."):
    g1, g2 = st.columns(2)
    for col, fname, cap in [
        (g1, "artist_long_tail.png", "Last.fm-2k: most artists had a single listener — near-unrecommendable."),
        (g2, "lorenz_popularity.png", "Last.fm-2k: an even steeper Lorenz curve (Gini 0.73)."),
    ]:
        path = FIG_DIR / fname
        if path.exists():
            col.image(str(path), caption=cap, width="stretch")
    st.caption("That artificial shape is exactly why we pivoted to 360K — and why the model ranking changed "
               "once real data arrived.")
