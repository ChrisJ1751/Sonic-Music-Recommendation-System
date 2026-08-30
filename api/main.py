"""FastAPI service for the artist recommender (Last.fm-360K).

On startup it loads the active dataset (data_config -> the 360K core), fits the
served model, **EASE**, the linear item-item autoencoder that won the model
comparison on this data, on ALL interactions (caching its weight matrix B to
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
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from contextlib import asynccontextmanager  # noqa: E402
from typing import Literal  # noqa: E402

from fastapi import FastAPI, HTTPException, Query, Response  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from api.logging_config import RequestLoggingMiddleware, configure_logging  # noqa: E402
from src import serving  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("api")
configure_logging()   # after get_logger, so its text handlers are replaced by JSON

# Where the Streamlit demo lives. Defaults to the local dev server; deployments
# set APP_URL (the Dockerfile points it at the live Space). Same pattern as
# REPORT_URL in app/views/overview.py.
APP_URL = os.environ.get("APP_URL", "http://localhost:8501")

# Public, read-only, unauthenticated GET API -- permissive CORS is appropriate
# and lets anyone call it from a browser. Narrow it with a comma-separated
# CORS_ALLOW_ORIGINS. Credentials are never allowed (nor can they be, with "*").
CORS_ALLOW_ORIGINS = [
    o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()
]

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Re-apply: uvicorn installs its own handlers after this module is imported,
    # so this second call is the one that wins in the container.
    configure_logging()
    started = time.perf_counter()
    STATE["reco"] = serving.load_state()
    state = STATE["reco"]
    logger.info("model loaded", extra={
        "event": "startup",
        "dataset": state.dataset,
        "model": "EASE",
        "n_users": state.n_users,
        "n_artists": state.n_artists,
        "load_seconds": round(time.perf_counter() - started, 2),
    })
    yield
    STATE.clear()


app = FastAPI(
    title="Sonic, Last.fm Artist Recommender",
    version="2.0.0",
    lifespan=lifespan,
    description=(
        "Serving layer for the Last.fm-360K artist recommender. Served model: "
        "**EASE** (Steck 2019), a linear item-item autoencoder.\n\n"
        "IDs are matrix indices: `user_id` is a user row, `artist_id` an artist "
        "column. A `user_id` past the end of the matrix falls back to popularity "
        "rather than 404ing.\n\n"
        "Every response carries an `X-Request-ID` (echoed if you supply one)."
    ),
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


Strategy = Literal["ease", "ease+mmr", "cold_start_popularity"]


class Recommendation(BaseModel):
    artist_id: int = Field(description="Artist column index in the interaction matrix.")
    name: str
    score: float = Field(description="EASE score; popularity count for the cold-start fallback.")


class RecommendationResponse(BaseModel):
    user_id: int
    strategy: Strategy = Field(
        description="`ease` for a known user, `ease+mmr` when diversity > 0, "
                    "`cold_start_popularity` when the user is not in the matrix."
    )
    k: int
    recommendations: list[Recommendation]


class SimilarArtistsResponse(BaseModel):
    artist_id: int
    name: str
    k: int
    similar: list[Recommendation]


class HealthResponse(BaseModel):
    """Readiness, not just liveness -- `status` is `ok` only once the model is loaded."""

    status: Literal["ok", "loading"]
    dataset: str | None
    model: str
    n_users: int
    n_artists: int


class ArtistRef(BaseModel):
    artist_id: int
    name: str


class PopularArtistsResponse(BaseModel):
    artists: list[ArtistRef]


class SampleUser(BaseModel):
    user_id: int
    top_artist: str


class SampleUsersResponse(BaseModel):
    users: list[SampleUser]


class TopArtist(BaseModel):
    artist_id: int
    name: str
    plays: int


class UserProfileResponse(BaseModel):
    user_id: int
    in_dataset: bool = Field(description="False for a user_id outside the matrix; still a 200.")
    n_artists: int | None = Field(default=None, description="Absent when in_dataset is false.")
    top_artists: list[TopArtist]


class AboutResponse(BaseModel):
    """Reported results + methodology. Static; the single source of truth is
    `src.serving.about_payload()`, which the Streamlit app renders too."""

    model: dict
    headline: list[dict]
    leaderboard: list[dict]
    significance: str
    beyond_accuracy: list[dict]
    cutoff_curve: dict
    methodology: list[dict]
    pivot: str
    stack: list[str]


def _reco() -> serving.RecoState:
    """The loaded model, or a clean 503 if startup has not finished.

    Without this the KeyError surfaces as a 500 with a stack trace, which reads
    as a bug rather than as "not ready yet", and gives a load balancer nothing
    actionable to retry on.
    """
    state = STATE.get("reco")
    if state is None:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Retry shortly.",
            headers={"Retry-After": "10"},
        )
    return state


_LANDING = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sonic API, Last.fm Recommender</title>
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
  <h1>Sonic, recommender API</h1>
  <p>This is the <b>FastAPI serving layer</b> for the Last.fm-360K artist
  recommender (served model: <b>EASE</b>). The interactive demo, results,
  charts, and the live recommender, lives in the <b>Streamlit app</b>.</p>
  <p><a href="__APP_URL__">Open the app</a>
     <a class="ghost" href="/docs">API explorer (/docs)</a>
     <a class="ghost" href="/about">Results (/about)</a></p>
  <p style="margin-top:18px">Quick call:
  <code>GET /recommendations/{user_id}?k=10&amp;diversity=0.3</code></p>
</div></body></html>"""

_LANDING = _LANDING.replace("__APP_URL__", APP_URL)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing() -> str:
    return _LANDING


@app.get("/health", response_model=HealthResponse,
         responses={503: {"description": "Model not loaded yet."}})
def health(response: Response) -> HealthResponse:
    """Readiness probe.

    Returns 503 until the model is actually loaded. Reporting 200 during startup
    would tell an orchestrator to route traffic at an instance that cannot serve
    a single recommendation.
    """
    s = STATE.get("reco")
    if s is None:
        response.status_code = 503
        response.headers["Retry-After"] = "10"
        return HealthResponse(status="loading", dataset=None, model="EASE",
                              n_users=0, n_artists=0)
    return HealthResponse(status="ok", dataset=s.dataset, model="EASE",
                          n_users=s.n_users, n_artists=s.n_artists)


@app.get("/about", response_model=AboutResponse)
def about() -> AboutResponse:
    """Project results + methodology (single source of truth: src.serving)."""
    return AboutResponse(**serving.about_payload())


@app.get("/popular-artists", response_model=PopularArtistsResponse)
def popular_artists(n: int = Query(50, ge=1, le=500)) -> PopularArtistsResponse:
    """Most-listened artists, ranked by distinct listeners."""
    arts = serving.popular_artists(_reco(), n)
    return PopularArtistsResponse(
        artists=[ArtistRef(artist_id=a["artist_id"], name=a["name"]) for a in arts]
    )


@app.get("/sample-users", response_model=SampleUsersResponse)
def sample_users(n: int = Query(6, ge=1, le=24)) -> SampleUsersResponse:
    """A deterministic set of users with rich histories, for demo quick-picks."""
    return SampleUsersResponse(users=serving.sample_users(_reco(), n))


@app.get("/users/{user_id}", response_model=UserProfileResponse)
def user_profile(user_id: int, k: int = Query(12, ge=1, le=50)) -> UserProfileResponse:
    """A user's most-played artists. Unknown users return 200 with `in_dataset: false`."""
    return UserProfileResponse(**serving.user_profile(_reco(), user_id, k))


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
    """'Fans also like', nearest artists in EASE's learned item-item weights."""
    out = serving.similar_artists(_reco(), artist_id, k)
    if out is None:
        raise HTTPException(status_code=404, detail=f"Unknown artist_id {artist_id}.")
    return SimilarArtistsResponse(**out)
