"""Offline GEE batch pipeline helpers (server-side reduceRegions/export pattern).

This module intentionally avoids runtime `.getInfo()` fetches for online API requests.
It provides deterministic specs/helpers that an offline job can use to:
1) build point FeatureCollections for a grid/time bucket,
2) run reduceRegions server-side over Earth Engine images,
3) export batch outputs (GCS/BigQuery),
4) serve API from exported snapshots via precomputed providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class BatchGridSpec:
    """Deterministic grid specification for offline EarthState extraction."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    step_deg: float


def normalize_bucket_start(dt: datetime, bucket_minutes: int) -> datetime:
    """Normalize timestamp to deterministic UTC bucket start."""
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be positive")
    dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    out = dt_utc.replace(second=0, microsecond=0)
    minute = out.minute - (out.minute % bucket_minutes)
    return out.replace(minute=minute)


def generate_grid_points(spec: BatchGridSpec) -> list[tuple[float, float]]:
    """Generate deterministic lat/lon grid points for batch FeatureCollection rows."""
    if spec.step_deg <= 0:
        raise ValueError("step_deg must be positive")

    lat = spec.lat_min
    out: list[tuple[float, float]] = []
    while lat <= spec.lat_max + 1e-9:
        lon = spec.lon_min
        while lon <= spec.lon_max + 1e-9:
            out.append((round(lat, 6), round(lon, 6)))
            lon += spec.step_deg
        lat += spec.step_deg
    return out


def build_export_description(bucket_start: datetime, dataset: str) -> str:
    """Build deterministic export task description names."""
    ts = bucket_start.astimezone(timezone.utc).strftime("%Y%m%dT%H%MZ")
    return f"skycolor_{dataset}_{ts}"
