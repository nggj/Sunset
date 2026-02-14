"""Integration test for mock ingest providers and signature kernel."""

from __future__ import annotations

from datetime import datetime, timezone

from skycolor_locator.ingest.mock_providers import MockEarthStateProvider, MockSurfaceProvider
from skycolor_locator.signature.core import compute_color_signature


def test_mock_providers_can_generate_signature_end_to_end() -> None:
    """Mock providers should support one end-to-end signature generation run."""
    dt = datetime(2024, 5, 12, 9, 0, tzinfo=timezone.utc)
    lat = 37.5665
    lon = 126.978

    earth = MockEarthStateProvider()
    surface = MockSurfaceProvider()

    atmos = earth.get_atmosphere_state(dt, lat, lon)
    surface_state = surface.get_surface_state(lat, lon)

    signature = compute_color_signature(
        dt=dt,
        lat=lat,
        lon=lon,
        atmos=atmos,
        surface=surface_state,
        config={"bins": 24, "n_az": 24, "n_el": 12},
    )

    assert len(signature.sky_hue_hist) == 24
    assert len(signature.ground_hue_hist) == 24
    assert len(signature.signature) == 48
    assert abs(sum(signature.sky_hue_hist) - 1.0) < 1e-6
    assert abs(sum(signature.ground_hue_hist) - 1.0) < 1e-6
