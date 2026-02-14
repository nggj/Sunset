"""API tests for signature/search endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def _client() -> object:
    """Create FastAPI TestClient with optional dependency guards."""
    pytest.importorskip("fastapi")
    testclient_module = pytest.importorskip("fastapi.testclient")
    from skycolor_locator.api.app import app

    return testclient_module.TestClient(app)


def test_signature_endpoint_returns_contract_payload() -> None:
    """`POST /signature` should return a contract-compatible payload."""
    client = _client()

    response = client.post(
        "/signature",
        json={
            "time_utc": datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc).isoformat(),
            "lat": 37.5665,
            "lon": 126.9780,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "hue_bins" in body
    assert "sky_hue_hist" in body
    assert "ground_hue_hist" in body
    assert "signature" in body
    assert len(body["signature"]) == 2 * len(body["hue_bins"])


def test_search_endpoint_reuses_cached_index_for_same_key() -> None:
    """`POST /search` should reuse cached index when request key is unchanged."""
    client = _client()

    payload = {
        "target_time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
        "target_lat": 37.5665,
        "target_lon": 126.9780,
        "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
        "grid_spec": {
            "lat_min": -20.0,
            "lat_max": 20.0,
            "lon_min": 100.0,
            "lon_max": 140.0,
            "step_deg": 20.0,
            "max_points": 20,
        },
        "top_k": 3,
    }

    first = client.post("/search", json=payload)
    second = client.post("/search", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200

    first_body = first.json()
    second_body = second.json()

    assert first_body["cache_hit"] is False
    assert second_body["cache_hit"] is True
    assert first_body["build_count"] == 1
    assert second_body["build_count"] == 1


def test_search_endpoint_returns_ranked_candidates_with_grid_inputs() -> None:
    """`POST /search` should return candidate keys from request grid."""
    client = _client()

    signature_response = client.post(
        "/signature",
        json={
            "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
            "lat": 10.0,
            "lon": 120.0,
        },
    )
    target_signature = signature_response.json()["signature"]

    search_response = client.post(
        "/search",
        json={
            "target_signature": target_signature,
            "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
            "grid_spec": {
                "lat_min": 0.0,
                "lat_max": 20.0,
                "lon_min": 110.0,
                "lon_max": 130.0,
                "step_deg": 10.0,
                "max_points": 20,
            },
            "top_k": 3,
            "filters": {"surface_classes": ["urban", "land", "ocean"]},
        },
    )

    assert search_response.status_code == 200
    body = search_response.json()
    assert len(body["candidates"]) == 3
    assert {"key", "distance"}.issubset(body["candidates"][0])
    assert "lat=" in body["candidates"][0]["key"]
    assert "lon=" in body["candidates"][0]["key"]


def test_search_endpoint_metric_affects_cache_key() -> None:
    """Different search metrics should not reuse the same cached index entry."""
    client = _client()

    base_payload = {
        "target_time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
        "target_lat": 37.5665,
        "target_lon": 126.9780,
        "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
        "grid_spec": {
            "lat_min": -20.0,
            "lat_max": 20.0,
            "lon_min": 100.0,
            "lon_max": 140.0,
            "step_deg": 20.0,
            "max_points": 20,
        },
        "top_k": 2,
    }

    cosine_response = client.post("/search", json={**base_payload, "metric": "cosine"})
    circular_response = client.post(
        "/search", json={**base_payload, "metric": "circular_emd"}
    )

    assert cosine_response.status_code == 200
    assert circular_response.status_code == 200

    cosine_body = cosine_response.json()
    circular_body = circular_response.json()

    assert cosine_body["cache_hit"] is False
    assert circular_body["cache_hit"] is False
    assert circular_body["build_count"] == cosine_body["build_count"] + 1
