"""Sonic — a music recommender for a Spotify DS portfolio.

Entry point: defines the sidebar navigation. Page content lives in views/, and
all model logic is in src/serving.py (shared with the FastAPI service). Run with:

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Sonic — Music Recommender", layout="wide")

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
