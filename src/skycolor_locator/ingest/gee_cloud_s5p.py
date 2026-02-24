"""Optional Sentinel-5P cloud feature fetch helpers for GEE providers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from skycolor_locator.ingest.gee_client import config_from_env, init_ee

_S5P_CLOUD = "COPERNICUS/S5P/NRTI/L3_CLOUD"


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware")
    return dt.astimezone(timezone.utc)


def _ensure_ee(ee: Any | None) -> Any:
    if ee is not None:
        return ee
    return init_ee(config_from_env())


def fetch_s5p_cloud_features(
    ee: Any | None,
    dt_utc: datetime,
    geometry: Any,
    time_window_hours: int,
) -> dict[str, float] | None:
    """Fetch mean Sentinel-5P cloud features near dt_utc over geometry."""
    ee_obj = _ensure_ee(ee)
    dt = _to_utc(dt_utc)
    window = timedelta(hours=time_window_hours)

    start = ee_obj.Date((dt - window).isoformat())
    end = ee_obj.Date((dt + window).isoformat())
    ic = ee_obj.ImageCollection(_S5P_CLOUD).filterDate(start, end).filterBounds(geometry)

    target_ms = ee_obj.Date(dt.isoformat()).millis()

    def add_diff(img: Any) -> Any:
        return img.set("time_diff", img.date().millis().subtract(target_ms).abs())

    best = ic.map(add_diff).sort("time_diff").first()

    bands = [
        "cloud_fraction",
        "cloud_optical_depth",
        "cloud_top_height",
        "cloud_base_height",
        "cloud_top_pressure",
        "cloud_base_pressure",
    ]
    reduced = (
        best.select(bands)
        .reduceRegion(
            reducer=ee_obj.Reducer.mean(),
            geometry=geometry,
            scale=10_000,
            maxPixels=10_000_000,
            bestEffort=True,
        )
        .getInfo()
    )

    if not isinstance(reduced, dict):
        return None

    out: dict[str, float] = {}
    for band in bands:
        value = reduced.get(band)
        if value is not None:
            out[band] = float(value)
    return out or None
