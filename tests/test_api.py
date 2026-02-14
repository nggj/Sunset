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


def test_search_endpoint_returns_ranked_candidates() -> None:
    """`POST /search` should return top-k candidate list."""
    client = _client()

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
        json={"target_signature": target_signature, "top_k": 3},
    )

    assert search_response.status_code == 200
    body = search_response.json()
    assert len(body["candidates"]) == 3
    assert {"key", "distance"}.issubset(body["candidates"][0])
