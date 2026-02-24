"""Deterministic angular tiling helpers."""

from __future__ import annotations

from math import floor


def _tile_origin(coord: float, step_deg: float) -> float:
    return floor(coord / step_deg) * step_deg


def tile_bounds(lat: float, lon: float, step_deg: float) -> tuple[float, float, float, float]:
    """Return tile bounds as (lon_min, lat_min, lon_max, lat_max)."""
    lat_min = _tile_origin(lat, step_deg)
    lon_min = _tile_origin(lon, step_deg)
    return (lon_min, lat_min, lon_min + step_deg, lat_min + step_deg)


def tile_id_for(lat: float, lon: float, step_deg: float = 0.05) -> str:
    """Deterministically encode tile id from lat/lon and angular tile step."""
    lon_min, lat_min, _, _ = tile_bounds(lat, lon, step_deg)
    return f"step{step_deg:.4f}:lat{lat_min:.4f}:lon{lon_min:.4f}"


def tile_center(lat: float, lon: float, step_deg: float) -> tuple[float, float]:
    """Return tile center as (center_lat, center_lon) for a lat/lon tile membership."""
    lon_min, lat_min, _, _ = tile_bounds(lat, lon, step_deg)
    return (lat_min + (step_deg / 2.0), lon_min + (step_deg / 2.0))
