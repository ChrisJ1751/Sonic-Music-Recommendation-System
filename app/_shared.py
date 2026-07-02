"""Shared helpers for the Streamlit app: path bootstrap, cached model state,
live-computed dataset/beyond-accuracy stats, and consistent Plotly styling.
Imported by the entry point and every page in views/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import serving  # noqa: E402

# The report's figures/ is the committed, curated set (outputs/figures is gitignored),
# so the hosted app finds its embedded charts. Falls back to outputs/figures locally.
FIG_DIR = ROOT / "report" / "figures"
if not FIG_DIR.exists():
    FIG_DIR = ROOT / "outputs" / "figures"

# palette (matches .streamlit/config.toml + the notebook figures)
GREEN = "#1ed760"
PURPLE = "#a880ff"      # lightened from #8b5cf6 for better contrast on dark
BLUE = "#4aa3ff"
AMBER = "#f0a63c"
MUTED = "#6b7688"       # readable muted (bars/labels)
FAINT = "#3a4150"       # non-highlighted bars
GRID = "rgba(255,255,255,0.07)"
TEXT = "#eef1f6"


@st.cache_resource(show_spinner="Loading the Last.fm-360K model (once) ...")
def get_state() -> serving.RecoState:
    """Load the dataset + fit/load EASE. Cached for the whole server process."""
    return serving.load_state()


@st.cache_data
def get_about() -> dict:
    return serving.about_payload()


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * x.sum()))


@st.cache_data(show_spinner="Computing dataset statistics ...")
def data_stats() -> dict:
    """Live EDA over the loaded 360K matrix — genuinely the served data."""
    s = get_state()
    m = s.im.matrix.tocsr()
    n_users, n_items = m.shape
    upu = np.asarray((m > 0).sum(axis=1)).ravel()          # artists per user
    ipi = np.asarray((m > 0).sum(axis=0)).ravel()          # listeners per artist
    plays = m.data
    return {
        "n_users": int(n_users), "n_items": int(n_items), "nnz": int(m.nnz),
        "density": m.nnz / (n_users * n_items),
        "median_history": int(np.median(upu)), "mean_history": float(upu.mean()),
        "median_listeners": int(np.median(ipi)), "max_listeners": int(ipi.max()),
        "gini_listeners": _gini(ipi),
        "top1pct_share": float(np.sort(ipi)[::-1][:max(1, n_items // 100)].sum() / ipi.sum()),
        "artists_per_user": upu, "listeners_per_artist": ipi,
        "median_plays": int(np.median(plays)), "p99_plays": float(np.percentile(plays, 99)),
        "max_plays": int(plays.max()),
    }


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    """Apply the dark theme consistently to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=13),
        title_font=dict(size=15),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:2px'>"
        f"<div style='width:34px;height:34px;border-radius:10px;"
        f"background:linear-gradient(135deg,{GREEN},{PURPLE})'></div>"
        f"<span style='font-size:22px;font-weight:700'>{title}</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(subtitle)
