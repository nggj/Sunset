"""Tests for horizon model effects on signature sampling and flags."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from skycolor_locator.contracts import AtmosphereState, CameraProfile, SurfaceClass, SurfaceState
from skycolor_locator.signature.core import compute_color_signature
from skycolor_locator.view.horizon import FlatHorizonModel


@dataclass
class _ConstantHorizonModel:
    value_deg: float = 5.0

    def horizon_profile(self, lat: float, lon: float, az_bins: int) -> list[float]:
        return [self.value_deg] * az_bins

    def meta(self) -> dict[str, str]:
        return {"model": "constant"}


def _inputs() -> tuple[AtmosphereState, SurfaceState]:
    atmos = AtmosphereState(cloud_fraction=0.2, aerosol_optical_depth=0.1, total_ozone_du=300.0)
    surface = SurfaceState(surface_class=SurfaceClass.LAND, dominant_albedo=0.3)
    return atmos, surface


def test_constant_horizon_reduces_visible_sky_fraction_vs_flat() -> None:
    atmos, surface = _inputs()
    dt = datetime(2024, 5, 12, 12, 0, tzinfo=UTC)
    camera = CameraProfile(pitch_deg=0.0, vfov_deg=45.0)

    sig_flat = compute_color_signature(
        dt,
        37.5,
        126.9,
        atmos,
        surface,
        config={"camera_profile": camera, "horizon_model": FlatHorizonModel()},
    )
    sig_const = compute_color_signature(
        dt,
        37.5,
        126.9,
        atmos,
        surface,
        config={"camera_profile": camera, "horizon_model": _ConstantHorizonModel(5.0)},
    )

    assert sig_const.meta["sky_fraction"] < sig_flat.meta["sky_fraction"]


def test_horizon_can_occlude_low_sun() -> None:
    atmos, surface = _inputs()
    dt = datetime(2024, 3, 20, 6, 10, tzinfo=UTC)

    sig = compute_color_signature(
        dt,
        0.0,
        0.0,
        atmos,
        surface,
        config={"horizon_model": _ConstantHorizonModel(5.0)},
    )

    assert sig.meta["sun_occluded"] is True
    assert "sun_occluded" in sig.quality_flags
