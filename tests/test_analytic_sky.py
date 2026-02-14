"""Tests for analytic sky rendering baseline."""

from __future__ import annotations

from colorsys import rgb_to_hsv
from datetime import datetime, timezone

from skycolor_locator.contracts import AtmosphereState
from skycolor_locator.sky.analytic import render_sky_rgb


def _mean_saturation(rgb_grid: list[list[list[float]]]) -> float:
    sats: list[float] = []
    for row in rgb_grid:
        for pixel in row:
            _, sat, _ = rgb_to_hsv(pixel[0], pixel[1], pixel[2])
            sats.append(sat)
    return sum(sats) / len(sats)


def _mean_hue(rgb_grid: list[list[list[float]]]) -> float:
    hues: list[float] = []
    for row in rgb_grid:
        for pixel in row:
            h, s, _ = rgb_to_hsv(pixel[0], pixel[1], pixel[2])
            if s > 1e-6:
                hues.append(h)
    return sum(hues) / len(hues)


def test_render_sky_rgb_shape_and_meta() -> None:
    """Renderer should produce (n_el, n_az, 3)-shaped nested lists and metadata."""
    atmos = AtmosphereState(
        cloud_fraction=0.2,
        aerosol_optical_depth=0.12,
        total_ozone_du=300.0,
        visibility_km=20.0,
    )
    rgb, meta = render_sky_rgb(
        dt=datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc),
        lat=37.5665,
        lon=126.978,
        atmos=atmos,
        n_az=24,
        n_el=12,
    )

    assert len(rgb) == 12
    assert all(len(row) == 24 for row in rgb)
    assert all(len(pixel) == 3 for row in rgb for pixel in row)
    assert "turbidity" in meta
    assert "sun_elev_deg" in meta


def test_render_sky_rgb_value_range() -> None:
    """Rendered RGB values must stay within [0, 1]."""
    atmos = AtmosphereState(
        cloud_fraction=0.5,
        aerosol_optical_depth=0.25,
        total_ozone_du=320.0,
        visibility_km=10.0,
    )
    rgb, _ = render_sky_rgb(
        dt=datetime(2024, 9, 15, 6, 0, tzinfo=timezone.utc),
        lat=0.0,
        lon=0.0,
        atmos=atmos,
        n_az=18,
        n_el=10,
    )

    assert all(0.0 <= channel <= 1.0 for row in rgb for pixel in row for channel in pixel)


def test_cloud_increase_desaturates_sky() -> None:
    """Higher cloud fractions should trend toward lower mean saturation."""
    dt = datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc)

    clear_atmos = AtmosphereState(
        cloud_fraction=0.0,
        aerosol_optical_depth=0.1,
        total_ozone_du=300.0,
        visibility_km=30.0,
    )
    cloudy_atmos = AtmosphereState(
        cloud_fraction=0.9,
        aerosol_optical_depth=0.1,
        total_ozone_du=300.0,
        visibility_km=30.0,
    )

    clear_rgb, _ = render_sky_rgb(dt=dt, lat=35.0, lon=129.0, atmos=clear_atmos, n_az=24, n_el=12)
    cloudy_rgb, _ = render_sky_rgb(dt=dt, lat=35.0, lon=129.0, atmos=cloudy_atmos, n_az=24, n_el=12)

    assert _mean_saturation(cloudy_rgb) < _mean_saturation(clear_rgb)


def test_sky_chromaticity_changes_with_sun_elevation() -> None:
    """Average sky hue should differ between noon and near-sunset conditions."""
    atmos = AtmosphereState(
        cloud_fraction=0.1,
        aerosol_optical_depth=0.12,
        total_ozone_du=300.0,
        visibility_km=25.0,
    )

    noon_rgb, _ = render_sky_rgb(
        dt=datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc),
        lat=0.0,
        lon=0.0,
        atmos=atmos,
        n_az=24,
        n_el=12,
    )
    sunset_rgb, _ = render_sky_rgb(
        dt=datetime(2024, 3, 20, 18, 0, tzinfo=timezone.utc),
        lat=0.0,
        lon=0.0,
        atmos=atmos,
        n_az=24,
        n_el=12,
    )

    assert abs(_mean_hue(noon_rgb) - _mean_hue(sunset_rgb)) > 0.005
