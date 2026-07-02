"""FastAPI service for the artist recommender (Last.fm-360K).

On startup it loads the active dataset (data_config -> the 360K core), fits the
served model — **EASE**, the linear item-item autoencoder that won the model
comparison on this data — on ALL interactions (caching its weight matrix B to
disk), and also fits a small ALS model to supply item embeddings for the MMR
diversity control. Recommendations come from EASE; "fans also like" from EASE's
item-item weights; unknown users fall back to popularity.

IDs are matrix indices: `user_id` = user row, `artist_id` = artist column
(the 360K native ids are opaque hashes / names, so indices are the clean public
handle and the API is dataset-agnostic).

Run:  uvicorn api.main:app --port 8000   then open  http://127.0.0.1:8000/
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src import serving  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("api")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["reco"] = serving.load_state()
    yield
    STATE.clear()


app = FastAPI(title="Sonic — Last.fm Artist Recommender", version="2.0.0", lifespan=lifespan)


class Recommendation(BaseModel):
    artist_id: int
    name: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    strategy: str  # "ease" | "ease+mmr" | "cold_start_popularity"
    k: int
    recommendations: list[Recommendation]


class SimilarArtistsResponse(BaseModel):
    artist_id: int
    name: str
    k: int
    similar: list[Recommendation]


def _reco() -> serving.RecoState:
    return STATE["reco"]


_LANDING = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sonic API — Last.fm Recommender</title>
<style>
  body{margin:0;min-height:100vh;display:grid;place-items:center;
    background:radial-gradient(800px 400px at 70% -10%,rgba(139,92,246,.12),transparent 60%),#0c0d11;
    color:#eef1f6;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  .card{max-width:560px;padding:36px;border:1px solid #272d3a;border-radius:20px;
    background:#141720;box-shadow:0 8px 30px rgba(0,0,0,.35)}
  h1{margin:0 0 4px;font-size:22px}
  p{color:#8b93a4;font-size:14px;line-height:1.6}
  a{display:inline-block;margin:6px 10px 0 0;padding:9px 15px;border-radius:999px;
    background:#1ed760;color:#04220f;font-weight:700;font-size:13px;text-decoration:none}
  a.ghost{background:transparent;color:#eef1f6;border:1px solid #272d3a}
  code{background:#0c0d11;border:1px solid #272d3a;border-radius:6px;padding:2px 6px;font-size:12px}
</style></head><body><div class="card">
  <h1>Sonic — recommender API</h1>
  <p>This is the <b>FastAPI serving layer</b> for the Last.fm-360K artist
  recommender (served model: <b>EASE</b>). The interactive demo — results,
  charts, and the live recommender — lives in the <b>Streamlit app</b>.</p>
  <p><a href="http://localhost:8501">Open the app</a>
     <a class="ghost" href="/docs">API explorer (/docs)</a>
     <a class="ghost" href="/about">Results (/about)</a></p>
  <p style="margin-top:18px">Quick call:
  <code>GET /recommendations/{user_id}?k=10&amp;diversity=0.3</code></p>
</div></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing() -> str:
    return _LANDING


@app.get("/health")
def health() -> dict:
    s = STATE.get("reco")
    return {"status": "ok", "dataset": s.dataset if s else None, "model": "EASE",
            "n_users": s.n_users if s else 0, "n_artists": s.n_artists if s else 0}


@app.get("/about")
def about() -> dict:
    """Project results + methodology (single source of truth: src.serving)."""
    return serving.about_payload()


@app.get("/popular-artists")
def popular_artists(n: int = Query(50, ge=1, le=500)) -> dict:
    arts = serving.popular_artists(_reco(), n)
    return {"artists": [{"artist_id": a["artist_id"], "name": a["name"]} for a in arts]}


@app.get("/sample-users")
def sample_users(n: int = Query(6, ge=1, le=24)) -> dict:
    return {"users": serving.sample_users(_reco(), n)}


@app.get("/users/{user_id}")
def user_profile(user_id: int, k: int = Query(12, ge=1, le=50)) -> dict:
    return serving.user_profile(_reco(), user_id, k)


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def recommendations(
    user_id: int,
    k: int = Query(10, ge=1, le=100),
    diversity: float = Query(0.0, ge=0.0, le=1.0,
                             description="0 = pure relevance; higher = more diverse (MMR re-ranking)"),
) -> RecommendationResponse:
    """Top-k EASE recommendations for a user; popularity fallback if unknown.

    `diversity` > 0 re-ranks a wider EASE candidate pool with MMR (using ALS item
    embeddings) to trade a little accuracy for a more varied list.
    """
    out = serving.recommend(_reco(), user_id, k=k, diversity=diversity)
    if not out["recommendations"]:
        raise HTTPException(status_code=404, detail="No recommendations available.")
    return RecommendationResponse(**out)


@app.get("/similar-artists/{artist_id}", response_model=SimilarArtistsResponse)
def similar_artists(artist_id: int, k: int = Query(10, ge=1, le=100)) -> SimilarArtistsResponse:
    """'Fans also like' — nearest artists in EASE's learned item-item weights."""
    out = serving.similar_artists(_reco(), artist_id, k)
    if out is None:
        raise HTTPException(status_code=404, detail=f"Unknown artist_id {artist_id}.")
    return SimilarArtistsResponse(**out)
