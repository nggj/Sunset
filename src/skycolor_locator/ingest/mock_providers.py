"""Deterministic mock providers for local/offline MVP flows."""

from __future__ import annotations

from datetime import datetime, timezone
from math import cos, pi, sin

from skycolor_locator.contracts import (
    AtmosphereState,
    PeriodicSurfaceConstants,
    SurfaceClass,
    SurfaceState,
)
from skycolor_locator.ingest.cache import LRUCache
from skycolor_locator.ingest.interfaces import (
    EarthStateProvider,
    PeriodicConstantsProvider,
    SurfaceProvider,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value to [lower, upper]."""
    return max(lower, min(upper, value))


class MockEarthStateProvider(EarthStateProvider):
    """Deterministic atmosphere provider based on location/time heuristics."""

    def __init__(self, cache_size: int = 256) -> None:
        """Initialize provider with optional LRU cache."""
        self._cache: LRUCache[tuple[str, int, int, int], AtmosphereState] = LRUCache(cache_size)

    def get_atmosphere_state(self, dt: datetime, lat: float, lon: float) -> AtmosphereState:
        """Return deterministic mock atmosphere state."""
        if dt.tzinfo is None:
            raise ValueError("dt must be timezone-aware")

        dt_utc = dt.astimezone(timezone.utc)
        key = (dt_utc.isoformat(), int(lat * 100), int(lon * 100), dt_utc.timetuple().tm_yday)

        def factory() -> AtmosphereState:
            seasonal = 0.5 + 0.5 * sin(2.0 * pi * (dt_utc.timetuple().tm_yday / 365.0))
            diurnal = 0.5 + 0.5 * cos(2.0 * pi * ((dt_utc.hour + dt_utc.minute / 60.0) / 24.0))
            lat_factor = 1.0 - abs(lat) / 90.0
            lon_wave = 0.5 + 0.5 * sin((lon + 180.0) * pi / 180.0)

            cloud_fraction = _clamp(0.15 + 0.45 * diurnal + 0.2 * (1.0 - lat_factor), 0.0, 1.0)
            aerosol_optical_depth = _clamp(0.06 + 0.15 * (1.0 - lat_factor) + 0.12 * lon_wave, 0.02, 0.8)
            total_ozone_du = _clamp(260.0 + 90.0 * seasonal + 40.0 * (1.0 - lat_factor), 220.0, 450.0)
            visibility_km = _clamp(35.0 - 22.0 * cloud_fraction - 10.0 * aerosol_optical_depth, 4.0, 40.0)

            return AtmosphereState(
                cloud_fraction=cloud_fraction,
                aerosol_optical_depth=aerosol_optical_depth,
                total_ozone_du=total_ozone_du,
                visibility_km=visibility_km,
            )

        return self._cache.get(key, factory)


class MockSurfaceProvider(SurfaceProvider):
    """Deterministic surface provider with coarse lat/lon-based classes."""

    def __init__(self, cache_size: int = 512) -> None:
        """Initialize provider with optional LRU cache."""
        self._cache: LRUCache[tuple[int, int], SurfaceState] = LRUCache(cache_size)

    def get_surface_state(self, lat: float, lon: float) -> SurfaceState:
        """Return deterministic mock surface state using coarse geo heuristics."""
        key = (int(lat * 100), int(lon * 100))

        def factory() -> SurfaceState:
            abs_lat = abs(lat)
            # Rough ocean belt: mid-ocean longitudes in tropics/subtropics.
            if abs_lat < 35.0 and (-160.0 <= lon <= -110.0 or 120.0 <= lon <= 160.0):
                return SurfaceState(
                    surface_class=SurfaceClass.OCEAN,
                    dominant_albedo=0.08,
                    landcover_mix={"ocean": 0.9, "land": 0.1},
                )

            # Urban heuristic near dense longitude bands + temperate latitudes.
            if 20.0 <= abs_lat <= 55.0 and (-20.0 <= lon <= 60.0 or 100.0 <= lon <= 140.0):
                return SurfaceState(
                    surface_class=SurfaceClass.URBAN,
                    dominant_albedo=0.2,
                    landcover_mix={"urban": 0.6, "land": 0.2, "forest": 0.2},
                )

            if abs_lat > 60.0:
                return SurfaceState(
                    surface_class=SurfaceClass.SNOW,
                    dominant_albedo=0.7,
                    landcover_mix={"snow": 0.8, "land": 0.2},
                )

            if 15.0 <= lat <= 35.0 and -20.0 <= lon <= 60.0:
                return SurfaceState(
                    surface_class=SurfaceClass.DESERT,
                    dominant_albedo=0.45,
                    landcover_mix={"desert": 0.8, "land": 0.2},
                )

            return SurfaceState(
                surface_class=SurfaceClass.LAND,
                dominant_albedo=0.25,
                landcover_mix={"land": 0.5, "forest": 0.4, "urban": 0.1},
            )

        return self._cache.get(key, factory)


class MockPeriodicConstantsProvider(PeriodicConstantsProvider):
    """Deterministic periodic constants provider for offline/testing workflows."""

    def get_periodic_surface_constants(
        self, dt: datetime, lat: float, lon: float
    ) -> PeriodicSurfaceConstants:
        """Return mock periodic constants with empty priors and source metadata."""
        del lat, lon
        dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

        return PeriodicSurfaceConstants(
            tile_id="mock-tile",
            period_start_utc=dt_utc,
            period_end_utc=dt_utc,
            landcover_mix={},
            class_rgb={},
            meta={"source": "mock"},
        )
