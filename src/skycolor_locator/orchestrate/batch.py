"""Local batch orchestration for signature indexing and querying."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from skycolor_locator.contracts import ColorSignature
from skycolor_locator.index.bruteforce import BruteforceIndex
from skycolor_locator.ingest.interfaces import EarthStateProvider, SurfaceProvider
from skycolor_locator.signature.core import compute_color_signature


@dataclass(frozen=True)
class GridSpec:
    """Lat/lon grid specification for batch signature generation."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    step_deg: float
    max_points: int | None = None


def _frange_inclusive(start: float, stop: float, step: float) -> list[float]:
    """Build an inclusive floating-point range with deterministic rounding."""
    if step <= 0.0:
        raise ValueError("step must be positive.")
    if start > stop:
        raise ValueError("start must be <= stop.")

    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def generate_lat_lon_grid(spec: GridSpec) -> list[tuple[float, float]]:
    """Generate `(lat, lon)` points from the input grid specification."""
    lats = _frange_inclusive(spec.lat_min, spec.lat_max, spec.step_deg)
    lons = _frange_inclusive(spec.lon_min, spec.lon_max, spec.step_deg)
    point_count = len(lats) * len(lons)
    if spec.max_points is not None and point_count > spec.max_points:
        raise ValueError("grid points exceed max_points safety cap")
    return [(lat, lon) for lat in lats for lon in lons]


def build_signature_index(
    dt: datetime,
    spec: GridSpec,
    earth_provider: EarthStateProvider,
    surface_provider: SurfaceProvider,
    config: dict[str, Any] | None = None,
    mode: Literal["cosine", "dot"] = "cosine",
) -> tuple[BruteforceIndex, dict[str, ColorSignature]]:
    """Compute signatures on a grid and build a local brute-force index."""
    dt_utc = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    dt_utc = dt_utc.astimezone(timezone.utc)

    index = BruteforceIndex(mode=mode)
    key_to_signature: dict[str, ColorSignature] = {}

    keys: list[str] = []
    vectors: list[list[float]] = []
    for lat, lon in generate_lat_lon_grid(spec):
        atmos = earth_provider.get_atmosphere_state(dt_utc, lat, lon)
        surface = surface_provider.get_surface_state(lat, lon)
        signature = compute_color_signature(dt_utc, lat, lon, atmos, surface, config=config)

        key = f"lat={lat:.3f},lon={lon:.3f}"
        keys.append(key)
        vectors.append(signature.signature)
        key_to_signature[key] = signature

    index.add(keys, vectors)
    return index, key_to_signature
