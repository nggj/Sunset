"""Tests for SurfaceState class_rgb ground palette overrides."""

from __future__ import annotations

from datetime import datetime, timezone

from skycolor_locator.contracts import AtmosphereState, SurfaceClass, SurfaceState
from skycolor_locator.signature.core import compute_color_signature


def test_ground_histogram_changes_with_class_rgb_override() -> None:
    """Overriding class RGB should alter the computed ground hue histogram."""
    dt = datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc)
    atmos = AtmosphereState(cloud_fraction=0.25, aerosol_optical_depth=0.12, total_ozone_du=305.0)

    default_surface = SurfaceState(
        surface_class=SurfaceClass.URBAN,
        dominant_albedo=0.3,
        landcover_mix={"urban": 1.0},
    )
    override_surface = SurfaceState(
        surface_class=SurfaceClass.URBAN,
        dominant_albedo=0.3,
        landcover_mix={"urban": 1.0},
        class_rgb={"urban": [1.0, 0.0, 0.0]},
    )

    default_sig = compute_color_signature(dt, 35.0, 129.0, atmos, default_surface, {"bins": 24})
    override_sig = compute_color_signature(dt, 35.0, 129.0, atmos, override_surface, {"bins": 24})

    assert default_sig.ground_hue_hist != override_sig.ground_hue_hist
