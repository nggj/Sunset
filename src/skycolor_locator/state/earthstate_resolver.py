"""Store-first EarthState resolver with provider fallback and optional write-back."""

from __future__ import annotations

from datetime import UTC, datetime

from skycolor_locator.contracts import AtmosphereState
from skycolor_locator.geo.tiling import tile_center, tile_id_for
from skycolor_locator.ingest.interfaces import EarthStateProvider
from skycolor_locator.state.earthstate_store import (
    EarthStateKey,
    EarthStateRecord,
    EarthStateStore,
)
from skycolor_locator.time.bucketing import bucket_start_utc, to_utc


class EarthStateResolver:
    """Resolve AtmosphereState via store-first lookup with provider fallback."""

    def __init__(
        self,
        provider: EarthStateProvider,
        store: EarthStateStore | None,
        tile_step_deg: float,
        bucket_minutes: int,
        writeback: bool,
    ) -> None:
        self._provider = provider
        self._store = store
        self._tile_step_deg = tile_step_deg
        self._bucket_minutes = bucket_minutes
        self._writeback = writeback

    def get_record(self, dt_utc: datetime, lat: float, lon: float) -> EarthStateRecord | None:
        """Return stored record for datetime/location key, or None when unavailable."""
        if self._store is None:
            return None
        key = self._build_key(dt_utc, lat, lon)
        return self._store.get(key)

    def get_atmosphere_state(self, dt_utc: datetime, lat: float, lon: float) -> AtmosphereState:
        """Resolve atmosphere from store or fallback provider, with optional write-back."""
        dt_resolved = to_utc(dt_utc)
        key = self._build_key(dt_resolved, lat, lon)

        if self._store is not None:
            cached = self._store.get(key)
            if cached is not None:
                return cached.atmos

        center_lat, center_lon = tile_center(lat, lon, self._tile_step_deg)
        atmos = self._provider.get_atmosphere_state(key.bucket_start_utc, center_lat, center_lon)

        if self._store is not None and self._writeback:
            self._store.put(
                EarthStateRecord(
                    key=key,
                    lat_center=center_lat,
                    lon_center=center_lon,
                    atmos=atmos,
                    meta={"source": "resolver_fallback_provider"},
                    ingested_at_utc=datetime.now(UTC),
                )
            )

        return atmos

    def _build_key(self, dt_utc: datetime, lat: float, lon: float) -> EarthStateKey:
        bucket_start = bucket_start_utc(dt_utc, self._bucket_minutes)
        return EarthStateKey(
            bucket_start_utc=bucket_start,
            bucket_minutes=self._bucket_minutes,
            tile_id=tile_id_for(lat, lon, self._tile_step_deg),
        )
