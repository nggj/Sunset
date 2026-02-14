"""Tests for color signature computation kernel."""

from __future__ import annotations

from datetime import datetime, timezone

from skycolor_locator.contracts import AtmosphereState, SurfaceClass, SurfaceState
from skycolor_locator.signature.core import _allocate_ground_sample_counts, compute_color_signature


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


def test_quality_flag_is_night_present_below_horizon() -> None:
    """is_night must be present when sun elevation is below or on horizon."""
    dt = datetime(2024, 3, 20, 0, 0, tzinfo=timezone.utc)
    atmos = AtmosphereState(cloud_fraction=0.1, aerosol_optical_depth=0.1, total_ozone_du=300.0)
    surface = SurfaceState(surface_class=SurfaceClass.LAND, dominant_albedo=0.3)

    sig = compute_color_signature(dt, 0.0, 0.0, atmos, surface, {"bins": 24})

    assert "is_night" in sig.quality_flags


def test_quality_flag_low_sun_present_for_small_positive_elevation() -> None:
    """low_sun must be present when elevation is positive but less than threshold."""
    dt = datetime(2024, 3, 20, 6, 10, tzinfo=timezone.utc)
    atmos = AtmosphereState(cloud_fraction=0.1, aerosol_optical_depth=0.1, total_ozone_du=300.0)
    surface = SurfaceState(surface_class=SurfaceClass.LAND, dominant_albedo=0.3)

    sig = compute_color_signature(dt, 0.0, 0.0, atmos, surface, {"bins": 24})

    assert "low_sun" in sig.quality_flags


def test_quality_flag_cloudy_present_for_high_cloud_fraction() -> None:
    """cloudy must be present when cloud_fraction crosses threshold."""
    dt = datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc)
    atmos = AtmosphereState(cloud_fraction=0.8, aerosol_optical_depth=0.2, total_ozone_du=300.0)
    surface = SurfaceState(surface_class=SurfaceClass.LAND, dominant_albedo=0.3)

    sig = compute_color_signature(dt, 0.0, 0.0, atmos, surface, {"bins": 24})

    assert "cloudy" in sig.quality_flags


def test_quality_flags_preserve_existing_extra_flags() -> None:
    """Extra flags should remain available along with required spec-minimum flags."""
    dt = datetime(2024, 3, 20, 0, 0, tzinfo=timezone.utc)
    atmos = AtmosphereState(
        cloud_fraction=0.85,
        aerosol_optical_depth=0.9,
        total_ozone_du=300.0,
        missing_realtime=True,
        cloud_optical_depth=20.0,
    )
    surface = SurfaceState(surface_class=SurfaceClass.LAND, dominant_albedo=0.3)

    sig = compute_color_signature(dt, 0.0, 0.0, atmos, surface, {"bins": 24})

    assert "is_night" in sig.quality_flags
    assert "cloudy" in sig.quality_flags
    assert "missing_realtime" in sig.quality_flags
    assert "high_cloud" in sig.quality_flags
    assert "sun_below_horizon" in sig.quality_flags


def test_ground_sample_allocation_is_bins_independent() -> None:
    """Ground sampling count should stay fixed across different bin settings."""
    dt, atmos, surface = _inputs()

    sig_18 = compute_color_signature(
        dt, 35.0, 129.0, atmos, surface, {"bins": 18, "ground_samples": 2000}
    )
    sig_72 = compute_color_signature(
        dt, 35.0, 129.0, atmos, surface, {"bins": 72, "ground_samples": 2000}
    )

    assert sig_18.meta["ground_sample_count"] == 2000
    assert sig_72.meta["ground_sample_count"] == 2000
    assert len(sig_18.signature) == 36
    assert len(sig_72.signature) == 144
    assert abs(sum(sig_18.ground_hue_hist) - 1.0) < 1e-6
    assert abs(sum(sig_72.ground_hue_hist) - 1.0) < 1e-6


def test_allocate_ground_sample_counts_is_deterministic() -> None:
    """Sample allocation should be deterministic and preserve total sample count."""
    palette = {"land": 0.7, "forest": 0.3}

    counts_first = _allocate_ground_sample_counts(palette, 2000)
    counts_second = _allocate_ground_sample_counts(palette, 2000)

    assert counts_first == counts_second
    assert sum(counts_first.values()) == 2000
