"""Sonic, a music recommender built on Last.fm listening data.

Entry point: defines the sidebar navigation. Page content lives in views/, and
all model logic is in src/serving.py (shared with the FastAPI service). Run with:

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os

# Pin BLAS threads BEFORE anything imports numpy. src/serving.py does the same,
# but by the time Streamlit imports it numpy is already initialised, so the
# setdefault there is a no-op and OpenBLAS spins up a 12-thread pool -- which is
# what the "OpenBLAS is configured to use 12 threads" warning in the logs was.
# Determinism is a claim this project makes; make it true at the entry point.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Sonic, Music Recommender", layout="wide")

nav = st.navigation({
    "The project": [
        st.Page("views/overview.py", title="Overview", default=True),
        st.Page("views/data.py", title="The data"),
        st.Page("views/results.py", title="Models & results"),
        st.Page("views/methodology.py", title="Methodology & limitations"),
    ],
    "Try it live": [
        st.Page("views/recommendations.py", title="Recommendations"),
        st.Page("views/artist_radio.py", title="Artist radio"),
    ],
})
nav.run()
