"""HTTP contract tests for the recommendation API -- no model artifacts needed.

These run in CI on every push. They use the `api_client` fixture, which injects a
real `serving.RecoState` built from toy data (see conftest.py) and skips the
artifact-loading lifespan. `tests/test_api.py` still exercises the real 360K core
and EASE matrix, but is marked `integration` and skips where those are absent.

What this covers that the integration test cannot: it always runs. Until now the
deployed surface was effectively untested in automation, because the only API
tests skipped whenever `matrix.npz` was missing -- which is every CI run.
"""
from __future__ import annotations

import logging

import pytest

REC_FIELDS = {"artist_id", "name", "score"}


# --- health -------------------------------------------------------------

def test_health_reports_loaded_model(api_client):
    body = api_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "EASE"
    assert body["n_users"] > 0 and body["n_artists"] > 0


# --- recommendations ----------------------------------------------------

def test_recommendations_shape_and_ordering(api_client):
    body = api_client.get("/recommendations/0?k=10").json()
    assert body["strategy"] == "ease"
    assert body["user_id"] == 0 and body["k"] == 10
    recs = body["recommendations"]
    assert len(recs) == 10
    assert all(set(r) == REC_FIELDS for r in recs)
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True), "pure-relevance ranking must be descending"
    assert len({r["artist_id"] for r in recs}) == 10, "no duplicate artists"


def test_recommendations_never_return_already_played(api_client, synthetic_state):
    played = set(synthetic_state.Xbin.getrow(0).indices.tolist())
    assert played, "fixture user should have listening history"
    recs = api_client.get("/recommendations/0?k=20").json()["recommendations"]
    assert played.isdisjoint({r["artist_id"] for r in recs})


def test_diversity_switches_strategy_to_mmr(api_client):
    body = api_client.get("/recommendations/0?k=8&diversity=0.5").json()
    assert body["strategy"] == "ease+mmr"
    assert len(body["recommendations"]) == 8
    # MMR deliberately trades relevance order for variety, so scores are NOT
    # required to be descending here -- only finite and well-formed.
    assert all(isinstance(r["score"], float) for r in body["recommendations"])


def test_unknown_user_falls_back_to_popularity(api_client):
    body = api_client.get("/recommendations/99999999?k=5").json()
    assert body["strategy"] == "cold_start_popularity"
    assert len(body["recommendations"]) == 5


@pytest.mark.parametrize("query", ["k=0", "k=101", "k=-1", "diversity=1.5", "diversity=-0.1", "k=abc"])
def test_recommendation_query_validation(api_client, query):
    assert api_client.get(f"/recommendations/0?{query}").status_code == 422


# --- similar artists ----------------------------------------------------

def test_similar_artists_excludes_self(api_client):
    body = api_client.get("/similar-artists/0?k=6").json()
    assert body["artist_id"] == 0 and len(body["similar"]) == 6
    assert all(s["artist_id"] != 0 for s in body["similar"])
    assert all(set(s) == REC_FIELDS for s in body["similar"])


def test_similar_artists_unknown_id_is_404(api_client):
    assert api_client.get("/similar-artists/99999999").status_code == 404


# --- user profile -------------------------------------------------------

def test_user_profile_known_and_unknown(api_client):
    known = api_client.get("/users/0?k=5").json()
    assert known["in_dataset"] is True
    plays = [a["plays"] for a in known["top_artists"]]
    assert plays == sorted(plays, reverse=True)

    # An out-of-range user is a 200 with in_dataset false, NOT a 404 -- that is
    # the documented contract the Streamlit app relies on.
    unknown = api_client.get("/users/99999999")
    assert unknown.status_code == 200
    assert unknown.json()["in_dataset"] is False


# --- catalogue helpers --------------------------------------------------

def test_popular_artists_and_sample_users(api_client):
    artists = api_client.get("/popular-artists?n=5").json()["artists"]
    assert len(artists) == 5 and all(set(a) == {"artist_id", "name"} for a in artists)
    assert api_client.get("/popular-artists?n=0").status_code == 422

    users = api_client.get("/sample-users?n=4").json()["users"]
    assert len(users) == 4
    assert all({"user_id", "top_artist"} == set(u) for u in users)


def test_about_and_landing(api_client):
    about = api_client.get("/about").json()
    assert about["model"]["name"] == "EASE"
    leaderboard = about["leaderboard"]
    assert leaderboard[0]["model"] == "EASE" and leaderboard[0]["served"] is True
    assert leaderboard == sorted(leaderboard, key=lambda r: r["ndcg10"], reverse=True)

    landing = api_client.get("/")
    assert landing.status_code == 200 and "Sonic" in landing.text


# --- structured logging -------------------------------------------------

def test_request_id_is_minted_and_echoed(api_client):
    response = api_client.get("/health")
    request_id = response.headers.get("x-request-id")
    assert request_id and len(request_id) == 32


def test_inbound_request_id_is_preserved(api_client):
    supplied = "0123456789abcdef0123456789abcdef"
    response = api_client.get("/health", headers={"X-Request-ID": supplied})
    assert response.headers["x-request-id"] == supplied


def test_request_log_carries_structured_fields(api_client, caplog):
    with caplog.at_level(logging.INFO, logger="api.request"):
        api_client.get("/recommendations/0?k=3")

    records = [r for r in caplog.records if r.name == "api.request"]
    assert records, "middleware should log one line per request"
    record = records[-1]
    assert record.method == "GET"
    assert record.path == "/recommendations/0"
    assert record.query == "k=3"
    assert record.status == 200
    assert isinstance(record.duration_ms, float)
    assert len(record.request_id) == 32


def test_client_error_is_logged_at_warning(api_client, caplog):
    with caplog.at_level(logging.INFO, logger="api.request"):
        api_client.get("/similar-artists/99999999")
    record = [r for r in caplog.records if r.name == "api.request"][-1]
    assert record.status == 404
    assert record.levelno == logging.WARNING


# --- readiness / degraded operation -------------------------------------
#
# These pin the behaviour of an instance whose model has NOT loaded -- during
# startup, or after a failed load. Previously /health returned 200 "ok" with
# n_users=0 (so an orchestrator would route traffic at it) and every
# model-backed endpoint raised a bare KeyError as a 500.

@pytest.fixture()
def unready_client(monkeypatch):
    """TestClient against an app whose lifespan never populated STATE."""
    from fastapi.testclient import TestClient

    from api import main

    monkeypatch.setattr(main, "STATE", {})
    return TestClient(main.app, raise_server_exceptions=False)


def test_health_reports_503_until_the_model_is_loaded(unready_client):
    response = unready_client.get("/health")
    assert response.status_code == 503, "must not tell a load balancer it is ready"
    assert response.json()["status"] == "loading"
    assert response.headers.get("Retry-After") == "10"


@pytest.mark.parametrize("path", [
    "/recommendations/0?k=5",
    "/users/0",
    "/popular-artists?n=3",
    "/sample-users",
    "/similar-artists/0",
])
def test_model_endpoints_return_503_not_500_when_unready(unready_client, path):
    response = unready_client.get(path)
    assert response.status_code == 503, f"{path} should be retryable, not a crash"
    assert response.headers.get("Retry-After") == "10"


def test_about_still_serves_without_the_model(unready_client):
    """/about is static project metadata -- no reason for it to need the matrix."""
    assert unready_client.get("/about").status_code == 200


def test_landing_page_serves_without_the_model(unready_client):
    assert unready_client.get("/").status_code == 200


# --- API contract surface -----------------------------------------------

def test_every_endpoint_is_typed_in_the_openapi_schema(api_client):
    """A `dict` return type documents nothing. /docs is this project's shop
    window, so every 200 response must resolve to a named schema."""
    spec = api_client.get("/openapi.json").json()
    untyped = []
    for path, operations in spec["paths"].items():
        for operation in operations.values():
            schema = (operation.get("responses", {}).get("200", {})
                      .get("content", {}).get("application/json", {}).get("schema", {}))
            if "$ref" not in schema:
                untyped.append(path)
    assert not untyped, f"untyped 200 responses: {untyped}"


def test_strategy_is_a_closed_enum_not_a_free_string(api_client):
    spec = api_client.get("/openapi.json").json()
    strategy = spec["components"]["schemas"]["RecommendationResponse"]["properties"]["strategy"]
    allowed = strategy.get("enum") or strategy.get("allOf", [{}])[0].get("enum")
    if allowed is None:                                   # pydantic may $ref the Literal
        ref = strategy.get("allOf", [{}])[0].get("$ref", "").split("/")[-1]
        allowed = spec["components"]["schemas"][ref]["enum"]
    assert set(allowed) == {"ease", "ease+mmr", "cold_start_popularity"}


def test_cors_is_enabled_for_browser_clients(api_client):
    response = api_client.get("/health", headers={"Origin": "https://example.com"})
    assert response.headers.get("access-control-allow-origin") == "*"
    assert "X-Request-ID" in response.headers.get("access-control-expose-headers", "")
