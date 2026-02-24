"""Tests for backward-compatible named featurization behavior."""

from datetime import UTC, datetime

from skycolor_locator.contracts import AtmosphereState, SurfaceClass, SurfaceState
from skycolor_locator.ml.features import featurize


def test_featurize_respects_feature_name_order_and_missing_defaults_to_zero() -> None:
    """Requested feature_names order should be preserved with unknown/missing -> 0.0."""
    dt = datetime(2024, 5, 12, 12, 0, tzinfo=UTC)
    atmos = AtmosphereState(cloud_fraction=0.2, aerosol_optical_depth=0.1, total_ozone_du=300.0)
    surface = SurfaceState(surface_class=SurfaceClass.LAND, dominant_albedo=0.3)

    names = ["cloud_fraction", "does_not_exist", "cloud_fraction_sat", "lat"]
    features, out_names = featurize(dt, 37.5, 126.9, atmos, surface, feature_names=names)

    assert out_names == names
    assert features[0] == 0.2
    assert features[1] == 0.0
    assert features[2] == 0.0
    assert features[3] == 37.5
