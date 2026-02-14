"""API tests for signature/search endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def _client_and_app() -> tuple[object, object]:
    """Create FastAPI TestClient with optional dependency guards."""
    pytest.importorskip("fastapi")
    testclient_module = pytest.importorskip("fastapi.testclient")
    from skycolor_locator.api.app import app

    return testclient_module.TestClient(app), app


def test_signature_endpoint_returns_contract_payload() -> None:
    """`POST /signature` should return a contract-compatible payload."""
    client, _ = _client_and_app()

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


def test_search_endpoint_returns_ranked_candidates() -> None:
    """`POST /search` should return top-k candidate list."""
    client, _ = _client_and_app()

    signature_response = client.post(
        "/signature",
        json={
            "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
            "lat": 37.5665,
            "lon": 126.9780,
        },
    )
    target_signature = signature_response.json()["signature"]

    search_response = client.post(
        "/search",
        json={
            "target_signature": target_signature,
            "top_k": 3,
            "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
        },
    )

    assert search_response.status_code == 200
    body = search_response.json()
    assert len(body["candidates"]) == 3
    assert {"key", "distance"}.issubset(body["candidates"][0])


def test_search_endpoint_accepts_filters_payload() -> None:
    """`POST /search` should accept optional filters object in request."""
    client, _ = _client_and_app()

    signature_response = client.post(
        "/signature",
        json={
            "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
            "lat": 37.5665,
            "lon": 126.9780,
        },
    )
    target_signature = signature_response.json()["signature"]

    search_response = client.post(
        "/search",
        json={
            "target_signature": target_signature,
            "top_k": 3,
            "filters": {"landcover": "urban"},
            "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
        },
    )

    assert search_response.status_code == 200
    assert "candidates" in search_response.json()


def test_search_reuses_cached_index_for_same_key() -> None:
    """Repeated identical /search requests should reuse cached index build."""
    client, app = _client_and_app()
    app.state.index_store.reset()

    payload = {
        "target": {
            "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
            "lat": 0.0,
            "lon": 110.0,
        },
        "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
        "grid_spec": {
            "lat_min": -5.0,
            "lat_max": 5.0,
            "lon_min": 105.0,
            "lon_max": 115.0,
            "step_deg": 5.0,
            "max_points": 100,
        },
        "top_k": 3,
    }

    first = client.post("/search", json=payload)
    second = client.post("/search", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert app.state.index_store.build_count == 1
