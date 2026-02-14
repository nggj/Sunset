"""Tests for batch orchestration utilities."""

from __future__ import annotations

from datetime import datetime, timezone

from skycolor_locator.ingest.mock_providers import MockEarthStateProvider, MockSurfaceProvider
from skycolor_locator.orchestrate.batch import GridSpec, build_signature_index, generate_lat_lon_grid


def test_generate_lat_lon_grid_inclusive() -> None:
    """Grid generation should include both range endpoints."""
    spec = GridSpec(lat_min=0.0, lat_max=10.0, lon_min=100.0, lon_max=110.0, step_deg=5.0)
    points = generate_lat_lon_grid(spec)

    assert points[0] == (0.0, 100.0)
    assert points[-1] == (10.0, 110.0)
    assert len(points) == 9


def test_build_signature_index_and_query() -> None:
    """Batch-generated index should be queryable with a matching signature."""
    dt = datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc)
    spec = GridSpec(lat_min=0.0, lat_max=10.0, lon_min=100.0, lon_max=110.0, step_deg=5.0)

    index, signatures = build_signature_index(
        dt=dt,
        spec=spec,
        earth_provider=MockEarthStateProvider(),
        surface_provider=MockSurfaceProvider(),
        config={"bins": 24, "n_az": 24, "n_el": 12},
    )

    key = "lat=5.000,lon=105.000"
    target_signature = signatures[key]
    results = index.query(target_signature.signature, top_k=1)

    assert results[0][0] == key
