"""API tests for signature/search endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pytest
from skycolor_locator.ingest.mock_providers import (
    MockEarthStateProvider,
    MockSurfaceProvider,
)
from skycolor_locator.ingest.periodic_precomputed_provider import (
    PrecomputedPeriodicConstantsProvider,
)
from skycolor_locator.ingest.precomputed_providers import (
    PrecomputedEarthStateProvider,
    PrecomputedSurfaceProvider,
)


def _client() -> object:
    """Create FastAPI TestClient with optional dependency guards."""
    pytest.importorskip("fastapi")
    testclient_module = pytest.importorskip("fastapi.testclient")
    from skycolor_locator.api.app import create_app

    return testclient_module.TestClient(create_app())


def _require_fastapi() -> None:
    """Skip tests that require FastAPI when dependency is unavailable."""
    pytest.importorskip("fastapi")


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


def test_create_app_uses_mock_providers_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_app should default to deterministic mock providers."""
    _require_fastapi()
    monkeypatch.delenv("SKYCOLOR_PROVIDER", raising=False)
    from skycolor_locator.api.app import create_app

    app = create_app()

    assert app.state.provider_mode == "mock"
    assert isinstance(app.state.earth_provider, MockEarthStateProvider)
    assert isinstance(app.state.surface_provider, MockSurfaceProvider)


def test_create_app_uses_precomputed_gee_providers_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """create_app should use precomputed snapshots for gee runtime mode."""
    _require_fastapi()
    from skycolor_locator.api.app import create_app

    earth_path = tmp_path / "earth.json"
    surface_path = tmp_path / "surface.json"
    earth_path.write_text(
        json.dumps(
            [
                {
                    "time_bucket_utc": "2024-05-12T09:00:00+00:00",
                    "lat": 37.566,
                    "lon": 126.978,
                    "cloud_fraction": 0.3,
                    "aerosol_optical_depth": 0.2,
                    "total_ozone_du": 300.0,
                }
            ]
        )
    )
    surface_path.write_text(
        json.dumps(
            [
                {
                    "lat": 37.566,
                    "lon": 126.978,
                    "surface_class": "urban",
                    "dominant_albedo": 0.2,
                    "landcover_mix": {"urban": 1.0},
                    "class_rgb": {},
                    "periodic_meta": {},
                }
            ]
        )
    )
    periodic_path = tmp_path / "periodic.json"
    periodic_path.write_text(
        json.dumps(
            [
                {
                    "tile_id": "step0.0500:lat37.5500:lon126.9500",
                    "period_start_utc": "2024-05-01T00:00:00+00:00",
                    "period_end_utc": "2024-05-31T23:59:59+00:00",
                    "landcover_mix": {"urban": 0.8, "land": 0.2},
                    "class_rgb": {"urban": [0.7, 0.7, 0.7]},
                    "meta": {"source": "s2_dynamic_world"},
                }
            ]
        )
    )

    monkeypatch.setenv("SKYCOLOR_GEE_EARTHSTATE_PATH", str(earth_path))
    monkeypatch.setenv("SKYCOLOR_GEE_SURFACE_PATH", str(surface_path))
    monkeypatch.setenv("SKYCOLOR_GEE_PERIODIC_PATH", str(periodic_path))

    app = create_app(provider_mode="gee")

    assert app.state.provider_mode == "gee"
    assert isinstance(app.state.earth_provider, PrecomputedEarthStateProvider)
    assert isinstance(app.state.surface_provider, PrecomputedSurfaceProvider)
    assert isinstance(app.state.periodic_constants_provider, PrecomputedPeriodicConstantsProvider)


def test_create_app_gee_mode_requires_snapshot_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_app gee mode should fail fast without precomputed dataset paths."""
    _require_fastapi()
    from skycolor_locator.api.app import create_app

    monkeypatch.delenv("SKYCOLOR_GEE_EARTHSTATE_PATH", raising=False)
    monkeypatch.delenv("SKYCOLOR_GEE_SURFACE_PATH", raising=False)

    with pytest.raises(ValueError, match="SKYCOLOR_GEE_EARTHSTATE_PATH"):
        create_app(provider_mode="gee")


def test_create_app_rejects_invalid_provider_mode() -> None:
    """create_app should reject unsupported provider mode values."""
    _require_fastapi()
    from skycolor_locator.api.app import create_app

    with pytest.raises(ValueError, match="SKYCOLOR_PROVIDER"):
        create_app(provider_mode="invalid")
