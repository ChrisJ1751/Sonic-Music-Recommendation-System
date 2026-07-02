"""Interactive recommender: pick a listener, see their taste and what EASE
suggests next. Bars are labelled progress columns so it's obvious what they mean.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from _shared import get_state, page_header

from src import serving

state = get_state()

page_header("Recommendations", "Pick a listener — see what they play, and what the model suggests next.")

# --- quick-pick sample users ---
samples = serving.sample_users(state, 6)
st.caption("Quick-pick a listener with a rich history:")
picks = st.columns(len(samples))
for col, u in zip(picks, samples, strict=True):
    if col.button(f"User {u['user_id']}\n{u['top_artist']}", use_container_width=True):
        st.session_state["uid"] = u["user_id"]

# --- controls ---
c1, c2, c3 = st.columns([1, 1, 2])
uid = int(c1.number_input("User #", min_value=0, max_value=state.n_users - 1,
                          value=int(st.session_state.get("uid", samples[0]["user_id"])), step=1))
k = c2.slider("Results", min_value=5, max_value=25, value=12)
diversity = c3.slider("Diversity (MMR re-ranking)", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                      help="0 = pure relevance; higher trades a little accuracy for a more varied list.")

prof = serving.user_profile(state, uid, k)
rec = serving.recommend(state, uid, k=k, diversity=diversity)

# --- taste snapshot ---
if prof["in_dataset"]:
    total_plays = int(state.im.matrix.getrow(uid).sum())
    top = prof["top_artists"][0]
    s1, s2, s3 = st.columns(3)
    s1.metric("Artists in history", f"{prof['n_artists']:,}")
    s2.metric("Total plays", f"{total_plays:,}")
    s3.metric("Top artist", top["name"], help=f"{top['plays']:,} plays")

badge = {"ease": "EASE (pure relevance)", "ease+mmr": "EASE + MMR diversity re-ranking",
         "cold_start_popularity": "cold-start (popularity fallback)"}[rec["strategy"]]

# Row height so ~k rows show without an inner scrollbar; "#" comes from the index.
def _table_height(n: int) -> int:
    return int(35 * (n + 1) + 3)


left, right = st.columns(2, gap="large")

with left:
    st.markdown("#### Listening profile")
    if not prof["in_dataset"]:
        st.warning("That user id is not in the dataset — they would get popularity-based cold-start recs.")
    else:
        st.caption(f"Their {prof['n_artists']} most-played artists · bar = plays, relative to their #1")
        df = pd.DataFrame(prof["top_artists"])
        df.index = range(1, len(df) + 1)
        st.dataframe(
            df[["name", "plays"]], use_container_width=True, height=_table_height(len(df)),
            column_config={
                "_index": st.column_config.NumberColumn("#", width="small"),
                "name": st.column_config.TextColumn("artist"),
                "plays": st.column_config.ProgressColumn(
                    "plays", format="%d", min_value=0, max_value=int(df["plays"].max()), width="medium"),
            },
        )

with right:
    st.markdown("#### Recommended for them")
    st.caption(f"strategy: **{badge}** · bar = recommendation strength, relative to the top pick")
    rdf = pd.DataFrame(rec["recommendations"])
    top_score = max(rdf["score"].max(), 1e-9)
    rdf["strength"] = 100.0 * rdf["score"] / top_score
    rdf.index = range(1, len(rdf) + 1)
    st.dataframe(
        rdf[["name", "strength"]], use_container_width=True, height=_table_height(len(rdf)),
        column_config={
            "_index": st.column_config.NumberColumn("#", width="small"),
            "name": st.column_config.TextColumn("artist (new to them)"),
            "strength": st.column_config.ProgressColumn(
                "match strength", format="%.0f%%", min_value=0.0, max_value=100.0, width="medium"),
        },
    )

st.caption("Drag **Diversity** above 0 to watch the list trade a little relevance for more variety — the "
           "same MMR lever the API exposes as its `diversity` parameter.")
