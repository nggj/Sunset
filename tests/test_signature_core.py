"""Tests for color signature computation kernel."""

from __future__ import annotations

from datetime import datetime, timezone

from skycolor_locator.contracts import AtmosphereState, SurfaceClass, SurfaceState
from skycolor_locator.signature.core import compute_color_signature


def _inputs() -> tuple[datetime, AtmosphereState, SurfaceState]:
    dt = datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc)
    atmos = AtmosphereState(
        cloud_fraction=0.25,
        aerosol_optical_depth=0.12,
        total_ozone_du=305.0,
        visibility_km=22.0,
    )
    surface = SurfaceState(
        surface_class=SurfaceClass.LAND,
        dominant_albedo=0.3,
        landcover_mix={"land": 0.7, "forest": 0.3},
    )
    return dt, atmos, surface


def test_compute_color_signature_histograms_normalized() -> None:
    """Sky and ground histograms should each sum to 1."""
    dt, atmos, surface = _inputs()
    sig = compute_color_signature(dt, 35.0, 129.0, atmos, surface, {"bins": 24})

    assert abs(sum(sig.sky_hue_hist) - 1.0) < 1e-6
    assert abs(sum(sig.ground_hue_hist) - 1.0) < 1e-6


def test_compute_color_signature_dimension() -> None:
    """Signature vector should have 2 * bins dimensions."""
    dt, atmos, surface = _inputs()
    sig = compute_color_signature(dt, 35.0, 129.0, atmos, surface, {"bins": 18})

    assert len(sig.signature) == 36
    assert len(sig.hue_bins) == 18


def test_compute_color_signature_deterministic() -> None:
    """Identical input should always produce identical output."""
    dt, atmos, surface = _inputs()
    cfg = {"bins": 20, "n_az": 24, "n_el": 12, "smooth_window": 3}

    first = compute_color_signature(dt, 35.0, 129.0, atmos, surface, cfg)
    second = compute_color_signature(dt, 35.0, 129.0, atmos, surface, cfg)

    assert first.signature == second.signature
    assert first.meta == second.meta
    assert first.uncertainty_score == second.uncertainty_score
