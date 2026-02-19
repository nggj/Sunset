"""Unit tests for periodic surface contracts."""

from __future__ import annotations

from datetime import UTC, datetime

from skycolor_locator.contracts import PeriodicSurfaceConstants, SurfaceClass, SurfaceState


def test_surface_state_backwards_compatible_without_class_rgb() -> None:
    """SurfaceState can still be instantiated without new optional fields."""
    state = SurfaceState(surface_class=SurfaceClass.LAND, dominant_albedo=0.25)

    assert state.class_rgb == {}
    assert state.periodic_meta == {}


def test_periodic_surface_constants_serializable_like_payload() -> None:
    """PeriodicSurfaceConstants accepts float dict payloads for landcover and class RGB."""
    constants = PeriodicSurfaceConstants(
        tile_id="tile-001",
        period_start_utc=datetime(2024, 1, 1, tzinfo=UTC),
        period_end_utc=datetime(2024, 12, 31, tzinfo=UTC),
        landcover_mix={"forest": 0.6, "urban": 0.4},
        class_rgb={"forest": [0.1, 0.5, 0.2], "urban": [0.4, 0.4, 0.4]},
    )

    assert all(isinstance(value, float) for value in constants.landcover_mix.values())
    assert all(
        isinstance(channel, float)
        for rgb in constants.class_rgb.values()
        for channel in rgb
    )
