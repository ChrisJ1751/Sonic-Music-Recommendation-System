"""API tests (Last.fm-360K + EASE). Skipped when the 360K core isn't built
(e.g. CI), since startup fits/loads the served model on the real matrix."""
import pytest
from fastapi.testclient import TestClient

from src.utils import PROCESSED_DIR

if not (PROCESSED_DIR / "lastfm360k" / "matrix.npz").exists():
    pytest.skip("360K core not built (run `python -m src.data_360k`)", allow_module_level=True)

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # one startup (loads matrix + EASE B + ALS) for the module
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok" and d["model"] == "EASE"
    assert d["n_users"] > 0 and d["n_artists"] > 0


def test_recommendations_known_user(client):
    r = client.get("/recommendations/0?k=10")
    assert r.status_code == 200
    d = r.json()
    assert d["strategy"] == "ease" and len(d["recommendations"]) == 10
    rec = d["recommendations"][0]
    assert set(rec) == {"artist_id", "name", "score"} and rec["name"]
    scores = [x["score"] for x in d["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_diversity_and_cold_start(client):
    r = client.get("/recommendations/0?k=8&diversity=0.5")
    assert r.status_code == 200 and r.json()["strategy"] == "ease+mmr"
    # a user index past the end falls back to popularity
    big = client.get("/recommendations/99999999?k=5").json()
    assert big["strategy"] == "cold_start_popularity" and len(big["recommendations"]) == 5
    assert client.get("/recommendations/0?k=0").status_code == 422


def test_user_profile_and_samples(client):
    d = client.get("/users/0?k=5").json()
    assert d["in_dataset"] is True and len(d["top_artists"]) == 5
    plays = [a["plays"] for a in d["top_artists"]]
    assert plays == sorted(plays, reverse=True)
    s = client.get("/sample-users?n=4").json()["users"]
    assert len(s) == 4 and all("user_id" in u and "top_artist" in u for u in s)


def test_similar_artists(client):
    aid = client.get("/popular-artists?n=1").json()["artists"][0]["artist_id"]
    d = client.get(f"/similar-artists/{aid}?k=6").json()
    assert len(d["similar"]) == 6
    assert all(s["artist_id"] != aid for s in d["similar"])
    assert client.get("/similar-artists/99999999").status_code == 404


def test_about(client):
    d = client.get("/about").json()
    assert d["model"]["name"] == "EASE"
    assert len(d["headline"]) == 4 and len(d["methodology"]) == 6
    # EASE must lead the leaderboard and be flagged as served
    lb = d["leaderboard"]
    assert lb[0]["model"] == "EASE" and lb[0]["served"] is True
    assert lb == sorted(lb, key=lambda r: r["ndcg10"], reverse=True)


def test_landing_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Sonic" in r.text
    assert "/docs" in r.text and "8501" in r.text  # points at API explorer + Streamlit app
