"""Tests for solar position calculations."""

from __future__ import annotations

from datetime import datetime, timezone

from skycolor_locator.astro.solar import solar_position


def test_solar_position_equinox_noon_equator_sanity() -> None:
    """Sun should be high near equatorial noon around the March equinox."""
    dt = datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc)
    sza_deg, saz_deg, sun_elev_deg = solar_position(dt, lat_deg=0.0, lon_deg=0.0)

    assert 0.0 <= sza_deg <= 10.0
    assert 80.0 <= sun_elev_deg <= 90.0
    assert 0.0 <= saz_deg < 360.0


def test_solar_position_elevation_zenith_consistency() -> None:
    """Elevation must be 90 - solar zenith by definition."""
    dt = datetime(2024, 6, 21, 9, 30, tzinfo=timezone.utc)
    sza_deg, _, sun_elev_deg = solar_position(dt, lat_deg=37.5665, lon_deg=126.978)

    assert abs(sun_elev_deg - (90.0 - sza_deg)) < 1e-9


def test_solar_position_deterministic() -> None:
    """The computation must be deterministic for identical inputs."""
    dt = datetime(2024, 12, 1, 0, 0, tzinfo=timezone.utc)
    first = solar_position(dt, lat_deg=-33.8688, lon_deg=151.2093)
    second = solar_position(dt, lat_deg=-33.8688, lon_deg=151.2093)

    assert first == second
