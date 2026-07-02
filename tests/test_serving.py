"""Unit tests for the shared inference core (src/serving.py). Skipped when the
360K core isn't built, since loading the state needs the real matrix."""
import pytest

from src.utils import PROCESSED_DIR

if not (PROCESSED_DIR / "lastfm360k" / "matrix.npz").exists():
    pytest.skip("360K core not built (run `python -m src.data_360k`)", allow_module_level=True)

from src import serving  # noqa: E402


@pytest.fixture(scope="module")
def state():
    return serving.load_state()


def test_recommend_known_user(state):
    out = serving.recommend(state, 0, k=10)
    assert out["strategy"] == "ease" and len(out["recommendations"]) == 10
    scores = [r["score"] for r in out["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    # recommendations must exclude the user's already-played artists
    played = {a["artist_id"] for a in serving.user_profile(state, 0, k=200)["top_artists"]}
    assert not ({r["artist_id"] for r in out["recommendations"]} & played)


def test_diversity_and_cold_start(state):
    assert serving.recommend(state, 0, k=8, diversity=0.5)["strategy"] == "ease+mmr"
    assert serving.recommend(state, state.n_users + 10, k=5)["strategy"] == "cold_start_popularity"


def test_similar_artists_and_bounds(state):
    aid = serving.popular_artists(state, 1)[0]["artist_id"]
    res = serving.similar_artists(state, aid, 6)
    assert len(res["similar"]) == 6 and all(s["artist_id"] != aid for s in res["similar"])
    assert serving.similar_artists(state, state.n_artists + 1, 6) is None


def test_about_payload_is_consistent():
    a = serving.about_payload()
    lb = a["leaderboard"]
    assert lb[0]["model"] == "EASE" and lb[0]["served"] is True
    assert lb == sorted(lb, key=lambda r: r["ndcg10"], reverse=True)
    # cutoff curve is monotone non-decreasing in k for both metrics
    assert a["cutoff_curve"]["ndcg"] == sorted(a["cutoff_curve"]["ndcg"])
    assert a["cutoff_curve"]["recall"] == sorted(a["cutoff_curve"]["recall"])
