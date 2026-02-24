"""Batch ingest for EarthState records into local SQLite store."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

from skycolor_locator.geo.tiling import tile_id_for
from skycolor_locator.ingest.factory import create_earth_provider
from skycolor_locator.state.earthstate_store import EarthStateKey, EarthStateRecord, SQLiteEarthStateStore
from skycolor_locator.time.bucketing import iter_bucket_starts, to_utc


@dataclass(frozen=True)
class IngestConfig:
    """Configuration for batch EarthState ingestion."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    tile_step_deg: float
    start_utc: datetime
    end_utc: datetime
    bucket_minutes: int
    provider_mode: str
    store_path: str
    max_tiles: int | None = None


def _frange(start: float, stop: float, step: float) -> list[float]:
    """Build deterministic inclusive floating range values."""
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 8))
        current += step
    return values


def generate_tile_centers(
    bounds: tuple[float, float, float, float],
    tile_step_deg: float,
) -> Iterator[tuple[float, float, str]]:
    """Yield tile centers and tile IDs for (lat_min, lat_max, lon_min, lon_max) bounds."""
    lat_min, lat_max, lon_min, lon_max = bounds
    if tile_step_deg <= 0.0:
        raise ValueError("tile_step_deg must be positive")
    if lat_min > lat_max:
        raise ValueError("lat_min must be <= lat_max")
    if lon_min > lon_max:
        raise ValueError("lon_min must be <= lon_max")

    lat_centers = _frange(lat_min, lat_max, tile_step_deg)
    lon_centers = _frange(lon_min, lon_max, tile_step_deg)
    for lat_center in lat_centers:
        for lon_center in lon_centers:
            yield (lat_center, lon_center, tile_id_for(lat_center, lon_center, tile_step_deg))


def ingest_earthstate(cfg: IngestConfig) -> dict[str, int]:
    """Ingest EarthState for configured tile/time grid and return ingest stats."""
    start_utc = to_utc(cfg.start_utc)
    end_utc = to_utc(cfg.end_utc)
    if end_utc <= start_utc:
        raise ValueError("end_utc must be greater than start_utc")
    if cfg.bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be positive")

    provider = create_earth_provider(cfg.provider_mode)
    store = SQLiteEarthStateStore(cfg.store_path)

    tile_entries = list(
        generate_tile_centers(
            (cfg.lat_min, cfg.lat_max, cfg.lon_min, cfg.lon_max),
            cfg.tile_step_deg,
        )
    )
    if cfg.max_tiles is not None:
        tile_entries = tile_entries[: cfg.max_tiles]

    total_written = 0
    missing_realtime_count = 0
    bucket_count = 0

    for bucket_start in iter_bucket_starts(start_utc, end_utc, cfg.bucket_minutes):
        bucket_count += 1
        records: list[EarthStateRecord] = []
        for lat_center, lon_center, tile_id in tile_entries:
            atmos = provider.get_atmosphere_state(bucket_start, lat_center, lon_center)
            if atmos.missing_realtime:
                missing_realtime_count += 1
            records.append(
                EarthStateRecord(
                    key=EarthStateKey(
                        bucket_start_utc=bucket_start,
                        bucket_minutes=cfg.bucket_minutes,
                        tile_id=tile_id,
                    ),
                    lat_center=lat_center,
                    lon_center=lon_center,
                    atmos=atmos,
                    meta={"provider_mode": cfg.provider_mode},
                    ingested_at_utc=datetime.now(UTC),
                )
            )
        store.bulk_put(records)
        total_written += len(records)

    return {
        "tiles_written": total_written,
        "buckets_written": bucket_count,
        "missing_realtime_count": missing_realtime_count,
    }
