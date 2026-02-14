"""Tests for residual histogram correction application."""

from __future__ import annotations

from datetime import datetime, timezone

from skycolor_locator.contracts import AtmosphereState, SurfaceClass, SurfaceState
from skycolor_locator.ml.features import featurize
from skycolor_locator.ml.residual_model import ResidualHistogramModel
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


def test_residual_apply_preserves_hist_constraints_and_determinism() -> None:
    """Residual correction should keep valid histograms and deterministic output."""
    dt, atmos, surface = _inputs()
    bins = 12

    baseline = compute_color_signature(dt, 35.0, 129.0, atmos, surface, {"bins": bins})
    x, feature_names = featurize(dt, 35.0, 129.0, atmos, surface)

    out_dim = 2 * bins
    weights = [[0.0] * out_dim for _ in feature_names]
    bias = [0.0] * out_dim
    bias[0] = 0.2
    bias[bins] = 0.15

    model = ResidualHistogramModel(
        version="residual_hist_ridge_v1",
        bins=bins,
        feature_names=feature_names,
        weights=weights,
        bias=bias,
    )

    corrected_a = model.apply_to_signature(baseline, x)
    corrected_b = model.apply_to_signature(baseline, x)

    assert corrected_a.signature == corrected_b.signature
    assert abs(sum(corrected_a.sky_hue_hist) - 1.0) < 1e-6
    assert abs(sum(corrected_a.ground_hue_hist) - 1.0) < 1e-6
    assert all(v >= 0.0 for v in corrected_a.sky_hue_hist)
    assert all(v >= 0.0 for v in corrected_a.ground_hue_hist)
    assert len(corrected_a.signature) == len(baseline.signature)
    assert corrected_a.meta["residual_applied"] is True
    assert corrected_a.meta["residual_model_version"] == "residual_hist_ridge_v1"
    assert "residual_applied" in corrected_a.quality_flags


def test_compute_color_signature_baseline_unchanged_when_residual_disabled() -> None:
    """Baseline output should remain unchanged unless residual is enabled."""
    dt, atmos, surface = _inputs()

    base_a = compute_color_signature(dt, 35.0, 129.0, atmos, surface, {"bins": 18})
    base_b = compute_color_signature(
        dt,
        35.0,
        129.0,
        atmos,
        surface,
        {"bins": 18, "apply_residual": False, "residual_model": None},
    )

    assert base_a.signature == base_b.signature
    assert base_a.meta == base_b.meta
    assert base_a.quality_flags == base_b.quality_flags
