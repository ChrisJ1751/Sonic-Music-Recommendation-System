"""Interactive recommender: pick a listener, see their taste and what EASE
suggests next.

The product surface. Controls sit in one panel, results in two cards beside it,
so changing a slider updates something visible without scrolling.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from _shared import card, get_state, page_header

from src import serving

state = get_state()

page_header("Recommendations", "Pick a listener — see what they play, and what the model suggests next.")


@st.cache_data(show_spinner=False)
def _quick_picks() -> list[dict]:
    """Deterministic (seeded), so caching changes nothing except the cost:
    uncached this rescanned the full 39,499 x 11,607 matrix on every rerun."""
    return serving.sample_users(state, 6)


def _clip(text: str, limit: int) -> str:
    """Artist names run from "beck" to "black rebel motorcycle club"; unclipped
    they wrap buttons to three lines and st.metric silently truncates them."""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


samples = _quick_picks()

# ---- control panel ----------------------------------------------------------
with st.container(border=True):
    st.caption("Quick-pick a listener with a rich history")
    picks = st.columns(len(samples))
    for col, user in zip(picks, samples, strict=True):
        label = _clip(user["top_artist"], 18)
        if col.button(f"User {user['user_id']}\n{label}", width="stretch",
                      help=f"{user['top_artist']} — their most-played artist"):
            st.session_state["uid"] = user["user_id"]

    c1, c2, c3 = st.columns([1, 1, 2])
    uid = int(c1.number_input("User #", min_value=0, max_value=state.n_users - 1,
                              value=int(st.session_state.get("uid", samples[0]["user_id"])), step=1))
    k = c2.slider("Results", min_value=5, max_value=25, value=12)
    diversity = c3.slider(
        "Diversity (MMR re-ranking)", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
        help="0 = pure relevance; higher trades a little accuracy for a more varied list.")

profile = serving.user_profile(state, uid, k)
rec = serving.recommend(state, uid, k=k, diversity=diversity)

STRATEGY_LABEL = {
    "ease": "EASE · pure relevance",
    "ease+mmr": "EASE + MMR diversity re-ranking",
    "cold_start_popularity": "cold start · popularity fallback",
}

# ---- taste snapshot ---------------------------------------------------------
if profile["in_dataset"]:
    total_plays = int(state.im.matrix.getrow(uid).sum())
    top = profile["top_artists"][0]
    s1, s2, s3, s4 = st.columns([1, 1, 1.7, 1.1])
    s1.metric("Artists in history", f"{profile['n_artists']:,}", border=True)
    s2.metric("Total plays", f"{total_plays:,}", border=True)
    s3.metric("Top artist", _clip(top["name"], 22),
              help=f"{top['name']} — {top['plays']:,} plays", border=True)
    s4.metric("Strategy", STRATEGY_LABEL[rec["strategy"]].split(" · ")[0],
              help=STRATEGY_LABEL[rec["strategy"]], border=True)

st.write("")


def _table_height(n: int) -> int:
    """Show ~n rows without an inner scrollbar; the header adds one row."""
    return int(35 * (n + 1) + 3)


left, right = st.columns(2, gap="medium")

with left:
    if not profile["in_dataset"]:
        with card("Listening profile"):
            st.warning("That user id is not in the dataset — they get popularity-based cold-start "
                       "recommendations, which is the documented fallback rather than an error.")
    else:
        with card("Listening profile",
                  f"Their top {len(profile['top_artists'])} of {profile['n_artists']} artists "
                  "· bar = plays, relative to their #1"):
            df = pd.DataFrame(profile["top_artists"])
            df.index = range(1, len(df) + 1)
            st.dataframe(
                df[["name", "plays"]], width="stretch", height=_table_height(len(df)),
                column_config={
                    "_index": st.column_config.NumberColumn("#", width="small"),
                    "name": st.column_config.TextColumn("artist"),
                    "plays": st.column_config.ProgressColumn(
                        "plays", format="%d", min_value=0,
                        max_value=int(df["plays"].max()), width="medium"),
                },
            )

with right:
    with card("Recommended for them", f"strategy: **{STRATEGY_LABEL[rec['strategy']]}**"):
        rdf = pd.DataFrame(rec["recommendations"])
        top_score = max(rdf["score"].max(), 1e-9)
        rdf["strength"] = 100.0 * rdf["score"] / top_score
        rdf.index = range(1, len(rdf) + 1)
        st.dataframe(
            rdf[["name", "strength"]], width="stretch", height=_table_height(len(rdf)),
            column_config={
                "_index": st.column_config.NumberColumn("#", width="small"),
                "name": st.column_config.TextColumn("artist (new to them)"),
                "strength": st.column_config.ProgressColumn(
                    "match strength", format="%.0f%%", min_value=0.0, max_value=100.0, width="medium"),
            },
        )

st.caption("Drag **Diversity** above 0 to watch the list trade a little relevance for more variety — the "
           "same MMR lever the API exposes as its `diversity` parameter.")
