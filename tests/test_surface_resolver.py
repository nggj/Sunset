"""Tests for SurfaceStateResolver periodic merge behavior."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from skycolor_locator.contracts import PeriodicSurfaceConstants, SurfaceClass, SurfaceState
from skycolor_locator.state.surface_resolver import SurfaceStateResolver


@dataclass
class _FixedSurfaceProvider:
    value: SurfaceState

    def get_surface_state(self, lat: float, lon: float) -> SurfaceState:
        return self.value


@dataclass
class _FixedPeriodicResolver:
    value: PeriodicSurfaceConstants

    def get_periodic_surface_constants(
        self,
        dt: datetime,
        lat: float,
        lon: float,
    ) -> PeriodicSurfaceConstants:
        return self.value


def test_surface_resolver_without_periodic_keeps_surface_unchanged() -> None:
    """Resolver should return same semantic surface values when periodic is disabled."""
    base = SurfaceState(
        surface_class=SurfaceClass.LAND,
        dominant_albedo=0.23,
        landcover_mix={"land": 1.0},
        class_rgb={"land": [0.4, 0.3, 0.2]},
        periodic_meta={},
    )
    resolver = SurfaceStateResolver(base_provider=_FixedSurfaceProvider(base), periodic=None)

    resolved = resolver.get_surface_state(datetime(2024, 5, 12, 9, tzinfo=UTC), 37.5, 126.9)
    assert resolved.surface_class == base.surface_class
    assert resolved.dominant_albedo == base.dominant_albedo
    assert resolved.landcover_mix == base.landcover_mix
    assert resolved.class_rgb == base.class_rgb
    assert resolved.periodic_meta == base.periodic_meta


def test_surface_resolver_with_periodic_merges_and_sets_meta() -> None:
    """Periodic constants should override mix/rgb and populate periodic metadata."""
    base = SurfaceState(
        surface_class=SurfaceClass.LAND,
        dominant_albedo=0.23,
        landcover_mix={"land": 1.0},
        class_rgb={"land": [0.4, 0.3, 0.2]},
        periodic_meta={"base": "keep"},
    )
    periodic = PeriodicSurfaceConstants(
        tile_id="step0.0500:lat37.5500:lon126.9500",
        period_start_utc=datetime(2024, 5, 1, tzinfo=UTC),
        period_end_utc=datetime(2024, 6, 1, tzinfo=UTC),
        landcover_mix={"urban": 0.7, "land": 0.3},
        class_rgb={"urban": [0.5, 0.5, 0.5]},
        meta={"source": "test", "revision": 2},
    )
    resolver = SurfaceStateResolver(
        base_provider=_FixedSurfaceProvider(base),
        periodic=_FixedPeriodicResolver(periodic),
    )

    resolved = resolver.get_surface_state(datetime(2024, 5, 12, 9, tzinfo=UTC), 37.5, 126.9)
    assert resolved.landcover_mix == periodic.landcover_mix
    assert resolved.class_rgb["land"] == [0.4, 0.3, 0.2]
    assert resolved.class_rgb["urban"] == [0.5, 0.5, 0.5]
    assert resolved.periodic_meta["tile_id"] == periodic.tile_id
    assert resolved.periodic_meta["period_start_utc"] == periodic.period_start_utc.isoformat()
    assert resolved.periodic_meta["period_end_utc"] == periodic.period_end_utc.isoformat()
    assert resolved.periodic_meta["source"] == "test"
    assert resolved.periodic_meta["revision"] == 2
