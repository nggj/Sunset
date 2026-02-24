"""Tests for store-first EarthState resolver behavior."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from skycolor_locator.contracts import AtmosphereState
from skycolor_locator.state.earthstate_resolver import EarthStateResolver
from skycolor_locator.state.earthstate_store import (
    EarthStateKey,
    EarthStateRecord,
    SQLiteEarthStateStore,
)
from skycolor_locator.time.bucketing import bucket_start_utc


@dataclass
class _RaisingProvider:
    def get_atmosphere_state(self, dt: datetime, lat: float, lon: float) -> AtmosphereState:
        raise RuntimeError("provider should not be called on store hit")


@dataclass
class _FixedProvider:
    atmos: AtmosphereState

    def get_atmosphere_state(self, dt: datetime, lat: float, lon: float) -> AtmosphereState:
        return self.atmos


def test_earthstate_resolver_store_hit_uses_cached_record(tmp_path) -> None:
    """Resolver should return cached atmosphere without calling fallback provider."""
    store = SQLiteEarthStateStore(str(tmp_path / "earthstate.db"))
    dt = datetime(2024, 5, 12, 9, 37, tzinfo=UTC)
    bucket_start = bucket_start_utc(dt, 60)
    cached_atmos = AtmosphereState(
        cloud_fraction=0.12,
        aerosol_optical_depth=0.09,
        total_ozone_du=302.0,
        missing_realtime=False,
    )
    store.put(
        EarthStateRecord(
            key=EarthStateKey(
                bucket_start_utc=bucket_start,
                bucket_minutes=60,
                tile_id="step0.0500:lat37.5500:lon126.9500",
            ),
            lat_center=37.575,
            lon_center=126.975,
            atmos=cached_atmos,
            meta={"provider": "seed"},
            ingested_at_utc=datetime(2024, 5, 12, 9, 40, tzinfo=UTC),
        )
    )

    resolver = EarthStateResolver(
        provider=_RaisingProvider(),
        store=store,
        tile_step_deg=0.05,
        bucket_minutes=60,
        writeback=True,
    )

    resolved = resolver.get_atmosphere_state(dt, 37.5665, 126.9780)
    assert resolved == cached_atmos


def test_earthstate_resolver_store_miss_fallback_and_writeback(tmp_path) -> None:
    """Resolver should fallback to provider and write back when cache miss occurs."""
    store = SQLiteEarthStateStore(str(tmp_path / "earthstate.db"))
    dt = datetime(2024, 5, 12, 9, 37, tzinfo=UTC)
    provided = AtmosphereState(
        cloud_fraction=0.44,
        aerosol_optical_depth=0.21,
        total_ozone_du=299.5,
        missing_realtime=True,
    )
    resolver = EarthStateResolver(
        provider=_FixedProvider(provided),
        store=store,
        tile_step_deg=0.05,
        bucket_minutes=60,
        writeback=True,
    )

    resolved = resolver.get_atmosphere_state(dt, 37.5665, 126.9780)
    assert resolved == provided

    record = resolver.get_record(dt, 37.5665, 126.9780)
    assert record is not None
    assert record.atmos == provided
