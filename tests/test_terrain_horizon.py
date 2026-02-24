"""Tests for DEM horizon helpers and terrain-aware signature adjustments."""

from __future__ import annotations

from datetime import datetime, timezone

from skycolor_locator.contracts import AtmosphereState, SurfaceClass, SurfaceState
from skycolor_locator.ingest.terrain_horizon import raycast_horizon_profile
from skycolor_locator.signature.core import compute_color_signature


def test_raycast_horizon_profile_detects_eastern_ridge() -> None:
    """Raycast horizon should report higher elevation angle toward ridge direction."""
    dem = [[0.0 for _ in range(9)] for _ in range(9)]
    # Build an eastern ridge (increasing columns) near observer row.
    for r in range(2, 7):
        dem[r][7] = 300.0

    horizon = raycast_horizon_profile(
        dem_m=dem,
        observer_row=4,
        observer_col=4,
        cell_size_m=30.0,
        max_distance_m=400.0,
        az_step_deg=45.0,
    )
    profile = horizon["horizon_profile_deg"]

    # az=90° (east) should be larger than az=270° (west).
    assert profile[2] > profile[6]


def test_terrain_horizon_can_occlude_sun_in_signature_meta() -> None:
    """Terrain horizon profile should expose sun occlusion metadata when configured."""
    dt = datetime(2024, 3, 20, 6, 10, tzinfo=timezone.utc)
    atmos = AtmosphereState(cloud_fraction=0.2, aerosol_optical_depth=0.1, total_ozone_du=300.0)
    surface = SurfaceState(
        surface_class=SurfaceClass.LAND,
        dominant_albedo=0.3,
        landcover_mix={"land": 1.0},
    )

    sig = compute_color_signature(
        dt,
        0.0,
        0.0,
        atmos,
        surface,
        {
            "bins": 24,
            "terrain_horizon_profile_deg": [25.0] * 72,
            "terrain_horizon_az_step_deg": 5.0,
        },
    )

    assert sig.meta["terrain_occluded_sun"] is True
    assert "terrain_occluded_sun" in sig.quality_flags
