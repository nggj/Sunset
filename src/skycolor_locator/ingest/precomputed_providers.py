"""Providers backed by precomputed EarthState/Surface snapshots.

These providers are intended for online API serving where runtime requests must avoid
Earth Engine synchronous `.getInfo()` calls. Data should be generated offline in batch
(e.g. reduceRegions + export), stored as files/table rows, and then loaded here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from skycolor_locator.contracts import AtmosphereState, SurfaceClass, SurfaceState
from skycolor_locator.ingest.cache import LRUCache
from skycolor_locator.ingest.interfaces import EarthStateProvider, SurfaceProvider


@dataclass(frozen=True)
class PrecomputedEarthRow:
    """One precomputed atmosphere row for a bucketed point/time key."""

    time_bucket_utc: str
    lat: float
    lon: float
    cloud_fraction: float
    aerosol_optical_depth: float
    total_ozone_du: float
    humidity: float | None = None
    visibility_km: float | None = None
    pressure_hpa: float | None = None
    cloud_optical_depth: float | None = None
    missing_realtime: bool = False


@dataclass(frozen=True)
class PrecomputedSurfaceRow:
    """One precomputed surface row for a point key."""

    lat: float
    lon: float
    surface_class: str
    dominant_albedo: float
    landcover_mix: dict[str, float]
    class_rgb: dict[str, list[float]]
    periodic_meta: dict[str, object]


def _normalize_time_bucket(dt: datetime, bucket_minutes: int) -> str:
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be positive")
    dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    bucket_start = dt_utc.replace(second=0, microsecond=0)
    minute = bucket_start.minute - (bucket_start.minute % bucket_minutes)
    bucket_start = bucket_start.replace(minute=minute)
    return bucket_start.isoformat()


def _point_key(lat: float, lon: float) -> tuple[int, int]:
    # 0.001° quantization keeps deterministic matching while tolerating float noise.
    return int(round(lat * 1000.0)), int(round(lon * 1000.0))


def _load_json_records(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [dict(item) for item in payload["rows"]]
    raise ValueError(f"Unsupported precomputed dataset format: {path}")


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return {str(k): v for k, v in value.items()}


def _as_float(value: object, field: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{field} must be numeric")


def _as_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return value


class PrecomputedEarthStateProvider(EarthStateProvider):
    """Atmosphere provider reading precomputed EarthState snapshots from disk."""

    def __init__(self, dataset_path: str, bucket_minutes: int = 60, cache_size: int = 512) -> None:
        self._bucket_minutes = bucket_minutes
        self._cache: LRUCache[tuple[str, int, int], AtmosphereState] = LRUCache(cache_size)
        rows = _load_json_records(Path(dataset_path))
        self._index: dict[tuple[str, int, int], AtmosphereState] = {}
        for row in rows:
            mapped = _as_mapping(row, "earth_row")
            bucket = str(mapped["time_bucket_utc"])
            lat = _as_float(mapped["lat"], "lat")
            lon = _as_float(mapped["lon"], "lon")
            key = (bucket, *_point_key(lat, lon))
            self._index[key] = AtmosphereState(
                cloud_fraction=_as_float(mapped["cloud_fraction"], "cloud_fraction"),
                aerosol_optical_depth=_as_float(
                    mapped["aerosol_optical_depth"], "aerosol_optical_depth"
                ),
                total_ozone_du=_as_float(mapped["total_ozone_du"], "total_ozone_du"),
                humidity=(
                    _as_float(mapped["humidity"], "humidity")
                    if mapped.get("humidity") is not None
                    else None
                ),
                visibility_km=(
                    _as_float(mapped["visibility_km"], "visibility_km")
                    if mapped.get("visibility_km") is not None
                    else None
                ),
                pressure_hpa=(
                    _as_float(mapped["pressure_hpa"], "pressure_hpa")
                    if mapped.get("pressure_hpa") is not None
                    else None
                ),
                cloud_optical_depth=(
                    _as_float(mapped["cloud_optical_depth"], "cloud_optical_depth")
                    if mapped.get("cloud_optical_depth") is not None
                    else None
                ),
                cloud_ice_fraction=(
                    _as_float(mapped["cloud_ice_fraction"], "cloud_ice_fraction")
                    if mapped.get("cloud_ice_fraction") is not None
                    else None
                ),
                cloud_effective_radius_um=(
                    _as_float(mapped["cloud_effective_radius_um"], "cloud_effective_radius_um")
                    if mapped.get("cloud_effective_radius_um") is not None
                    else None
                ),
                missing_realtime=bool(mapped.get("missing_realtime", False)),
            )

    def get_atmosphere_state(self, dt: datetime, lat: float, lon: float) -> AtmosphereState:
        """Return atmosphere state from precomputed index for the bucketed point/time key."""
        bucket = _normalize_time_bucket(dt, self._bucket_minutes)
        key = (bucket, *_point_key(lat, lon))

        def factory() -> AtmosphereState:
            value = self._index.get(key)
            if value is None:
                raise KeyError(
                    "Missing precomputed EarthState for "
                    f"bucket={bucket}, lat={lat:.3f}, lon={lon:.3f}"
                )
            return value

        return self._cache.get(key, factory)


class PrecomputedSurfaceProvider(SurfaceProvider):
    """Surface provider reading precomputed surface constants from disk."""

    def __init__(self, dataset_path: str, cache_size: int = 512) -> None:
        self._cache: LRUCache[tuple[int, int], SurfaceState] = LRUCache(cache_size)
        rows = _load_json_records(Path(dataset_path))
        self._index: dict[tuple[int, int], SurfaceState] = {}
        for row in rows:
            mapped = _as_mapping(row, "surface_row")
            cls = SurfaceClass(str(mapped["surface_class"]))
            lat = _as_float(mapped["lat"], "lat")
            lon = _as_float(mapped["lon"], "lon")
            key = _point_key(lat, lon)
            mix = _as_mapping(mapped.get("landcover_mix", {}), "landcover_mix")
            class_rgb_raw = _as_mapping(mapped.get("class_rgb", {}), "class_rgb")
            periodic_meta = _as_mapping(mapped.get("periodic_meta", {}), "periodic_meta")
            self._index[key] = SurfaceState(
                surface_class=cls,
                dominant_albedo=_as_float(mapped["dominant_albedo"], "dominant_albedo"),
                landcover_mix={str(k): _as_float(v, "landcover_mix value") for k, v in mix.items()},
                class_rgb={
                    str(k): [_as_float(c, "class_rgb channel") for c in _as_list(v, "class_rgb")]
                    for k, v in class_rgb_raw.items()
                },
                periodic_meta=dict(periodic_meta),
            )

    def get_surface_state(self, lat: float, lon: float) -> SurfaceState:
        """Return surface state from precomputed index for the quantized point key."""
        key = _point_key(lat, lon)

        def factory() -> SurfaceState:
            value = self._index.get(key)
            if value is None:
                raise KeyError(f"Missing precomputed surface state for lat={lat:.3f}, lon={lon:.3f}")
            return value

        return self._cache.get(key, factory)
