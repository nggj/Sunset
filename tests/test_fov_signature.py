"""Tests for camera FOV-aware signature behavior."""

from datetime import UTC, datetime

from skycolor_locator.contracts import AtmosphereState, CameraProfile, SurfaceClass, SurfaceState
from skycolor_locator.signature.core import compute_color_signature


def _inputs() -> tuple[datetime, AtmosphereState, SurfaceState]:
    dt = datetime(2024, 5, 12, 12, 0, tzinfo=UTC)
    atmos = AtmosphereState(cloud_fraction=0.2, aerosol_optical_depth=0.1, total_ozone_du=300.0)
    surface = SurfaceState(surface_class=SurfaceClass.LAND, dominant_albedo=0.3)
    return dt, atmos, surface


def test_camera_pitched_up_has_no_ground_flag_and_normalized_hists() -> None:
    dt, atmos, surface = _inputs()
    camera = CameraProfile(pitch_deg=85.0, vfov_deg=30.0)

    sig = compute_color_signature(
        dt,
        37.5,
        126.9,
        atmos,
        surface,
        config={"bins": 24, "camera_profile": camera},
    )

    assert "no_ground" in sig.quality_flags
    assert abs(sum(sig.sky_hue_hist) - 1.0) < 1e-6
    assert abs(sum(sig.ground_hue_hist) - 1.0) < 1e-6


def test_camera_pitched_down_has_no_sky_flag() -> None:
    dt, atmos, surface = _inputs()
    camera = CameraProfile(pitch_deg=-85.0, vfov_deg=30.0)

    sig = compute_color_signature(
        dt,
        37.5,
        126.9,
        atmos,
        surface,
        config={"bins": 24, "camera_profile": camera},
    )

    assert "no_sky" in sig.quality_flags
    assert abs(sum(sig.sky_hue_hist) - 1.0) < 1e-6
    assert abs(sum(sig.ground_hue_hist) - 1.0) < 1e-6
