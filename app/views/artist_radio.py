"""Artist radio — 'fans also like'. Nearest artists in EASE's learned
item-item space, symmetrised."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from _shared import get_state, page_header

from src import serving

state = get_state()

page_header("Artist radio",
            "Nearest artists in EASE's learned item-item space — the model's 'fans also like'.")


@st.cache_data(show_spinner=False)
def popular() -> list[dict]:
    return serving.popular_artists(state, 200)


artists = popular()
names = [a["name"] for a in artists]
default = names.index("radiohead") if "radiohead" in names else 0

choice = st.selectbox("Pick an artist (top 200 by listeners)", names, index=default)
picked = next(a for a in artists if a["name"] == choice)
rank = names.index(choice) + 1

m1, m2 = st.columns(2)
m1.metric("Distinct listeners", f"{picked['listeners']:,}")
m2.metric("Popularity rank", f"#{rank}", help="Among the top-200 most-listened artists in the catalogue.")

res = serving.similar_artists(state, picked["artist_id"], 12)
if res is None or not res["similar"]:
    st.info("No neighbours found for this artist.")
else:
    st.caption(f"Most similar to **{res['name']}** · similarity = symmetrised EASE item-item weight, "
               "relative to the closest neighbour")
    df = pd.DataFrame(res["similar"])
    top = max(df["score"].max(), 1e-9)
    df["match"] = 100.0 * df["score"] / top
    df.index = range(1, len(df) + 1)
    st.dataframe(
        df[["name", "match"]], use_container_width=True, height=int(35 * (len(df) + 1) + 3),
        column_config={
            "_index": st.column_config.NumberColumn("#", width="small"),
            "name": st.column_config.TextColumn("artist"),
            "match": st.column_config.ProgressColumn(
                "similarity", format="%.0f%%", min_value=0.0, max_value=100.0, width="medium"),
        },
    )
    st.caption("These are the artists whose listeners most overlap with the picked one, learned purely from "
               "co-listening — no genre tags or audio features involved.")
