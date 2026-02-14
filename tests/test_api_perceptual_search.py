"""API tests for perceptual_v1 search."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def _client() -> object:
    """Create isolated FastAPI test client."""
    pytest.importorskip("fastapi")
    testclient_module = pytest.importorskip("fastapi.testclient")
    from skycolor_locator.api.app import create_app

    return testclient_module.TestClient(create_app())


def test_search_endpoint_supports_perceptual_v1_generation() -> None:
    """`POST /search` should work with target point generation in perceptual_v1 mode."""
    client = _client()

    response = client.post(
        "/search",
        json={
            "vector_type": "perceptual_v1",
            "metric": "cosine",
            "target_time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
            "target_lat": 10.0,
            "target_lon": 120.0,
            "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
            "grid_spec": {
                "lat_min": 0.0,
                "lat_max": 10.0,
                "lon_min": 110.0,
                "lon_max": 120.0,
                "step_deg": 10.0,
                "max_points": 10,
            },
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["candidates"]) == 2
    assert "distance" in body["candidates"][0]


def test_search_endpoint_rejects_emd_with_perceptual_v1() -> None:
    """perceptual_v1 should reject EMD-family metrics."""
    client = _client()

    response = client.post(
        "/search",
        json={
            "vector_type": "perceptual_v1",
            "metric": "emd",
            "target_signature": [0.1, 0.2, 0.3, 0.4],
            "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
            "top_k": 1,
        },
    )

    assert response.status_code == 422


def test_signature_endpoint_rejects_residual_without_loaded_model() -> None:
    """`POST /signature` should return 422 when residual model is unavailable."""
    client = _client()

    response = client.post(
        "/signature",
        json={
            "time_utc": datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
            "lat": 10.0,
            "lon": 120.0,
            "apply_residual": True,
        },
    )

    assert response.status_code == 422
