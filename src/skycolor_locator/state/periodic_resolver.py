"""Periodic surface constants resolver with store-first and optional builder."""

from __future__ import annotations

from datetime import UTC, datetime

from skycolor_locator.contracts import PeriodicSurfaceConstants
from skycolor_locator.geo.tiling import tile_center, tile_id_for
from skycolor_locator.state.periodic_store import PeriodicKey, PeriodicRecord, PeriodicConstantsStore
from skycolor_locator.time.periods import period_month


class PeriodicConstantsResolver:
    """Resolve periodic constants via local store and optional S2 builder fallback."""

    def __init__(
        self,
        store: PeriodicConstantsStore | None,
        builder_enabled: bool,
        tile_step_deg: float,
        provider_mode: str,
        writeback: bool,
    ) -> None:
        self._store = store
        self._builder_enabled = builder_enabled
        self._tile_step_deg = tile_step_deg
        self._provider_mode = provider_mode.strip().lower()
        self._writeback = writeback

    def get_periodic_surface_constants(
        self,
        dt: datetime,
        lat: float,
        lon: float,
    ) -> PeriodicSurfaceConstants:
        """Return periodic constants from store, builder, or deterministic empty fallback."""
        period_start, period_end = period_month(dt)
        tile_id = tile_id_for(lat, lon, self._tile_step_deg)
        key = PeriodicKey(tile_id=tile_id, period_start_utc=period_start, period_end_utc=period_end)

        if self._store is not None:
            existing = self._store.get(key)
            if existing is not None:
                return existing.constants

        if self._builder_enabled and self._provider_mode == "gee":
            center_lat, center_lon = tile_center(lat, lon, self._tile_step_deg)
            from skycolor_locator.ingest.s2_periodic import S2PeriodicConstantsBuilder

            constants = S2PeriodicConstantsBuilder().build(
                period_start_utc=period_start,
                period_end_utc=period_end,
                lat=center_lat,
                lon=center_lon,
            )
            if self._store is not None and self._writeback:
                self._store.put(
                    PeriodicRecord(
                        key=key,
                        constants=constants,
                        meta={"source": "s2_builder"},
                        ingested_at_utc=datetime.now(UTC),
                    )
                )
            return constants

        return PeriodicSurfaceConstants(
            tile_id=tile_id,
            period_start_utc=period_start,
            period_end_utc=period_end,
            landcover_mix={},
            class_rgb={},
            meta={"source": "none"},
        )
