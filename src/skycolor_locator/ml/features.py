"""Deterministic feature extraction for residual histogram correction."""

from __future__ import annotations

from datetime import datetime, timezone
from math import cos, log1p, pi, sin

from skycolor_locator.astro.solar import solar_position
from skycolor_locator.contracts import AtmosphereState, SurfaceClass, SurfaceState
from skycolor_locator.sky.analytic import estimate_turbidity

_KNOWN_CLASSES: list[str] = [
    SurfaceClass.OCEAN.value,
    SurfaceClass.LAND.value,
    SurfaceClass.URBAN.value,
    SurfaceClass.SNOW.value,
    SurfaceClass.DESERT.value,
    SurfaceClass.FOREST.value,
]


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


DEFAULT_FEATURE_NAMES: list[str] = [
    "lat",
    "abs_lat",
    "day_of_year_sin",
    "day_of_year_cos",
    "sun_elev_deg",
    "sza_deg",
    "sin_saz_deg",
    "cos_saz_deg",
    "turbidity",
    "cloud_fraction",
    "log1p_cloud_optical_depth",
    "aerosol_optical_depth",
    "ozone_over_300",
    "dominant_albedo",
] + [f"surface_class_is_{name}" for name in _KNOWN_CLASSES] + [
    f"landcover_mix_{name}" for name in _KNOWN_CLASSES
]


_OPTIONAL_CLOUD_FIELDS: list[str] = [
    "cloud_fraction_low",
    "cloud_fraction_mid",
    "cloud_fraction_high",
    "cloud_optical_depth_low",
    "cloud_optical_depth_mid",
    "cloud_optical_depth_high",
    "cloud_fraction_sat",
    "cloud_optical_depth_sat",
    "cloud_top_height_m",
    "cloud_base_height_m",
    "cloud_top_pressure_hpa",
    "cloud_base_pressure_hpa",
]


def compute_feature_dict(
    dt: datetime,
    lat: float,
    lon: float,
    atmos: AtmosphereState,
    surface: SurfaceState,
) -> dict[str, float]:
    """Compute named feature dictionary including legacy and extended cloud features."""
    dt_utc = _to_utc(dt)
    day_of_year = dt_utc.timetuple().tm_yday
    day_theta = 2.0 * pi * day_of_year / 365.0

    sza_deg, saz_deg, sun_elev_deg = solar_position(dt_utc, lat, lon)
    saz_rad = saz_deg * pi / 180.0

    turbidity = estimate_turbidity(atmos.aerosol_optical_depth, atmos.visibility_km)
    cod = atmos.cloud_optical_depth if atmos.cloud_optical_depth is not None else 0.0

    feature_map: dict[str, float] = {
        "lat": float(lat),
        "abs_lat": abs(float(lat)),
        "day_of_year_sin": sin(day_theta),
        "day_of_year_cos": cos(day_theta),
        "sun_elev_deg": sun_elev_deg,
        "sza_deg": sza_deg,
        "sin_saz_deg": sin(saz_rad),
        "cos_saz_deg": cos(saz_rad),
        "turbidity": float(turbidity),
        "cloud_fraction": float(atmos.cloud_fraction),
        "log1p_cloud_optical_depth": log1p(max(0.0, float(cod))),
        "aerosol_optical_depth": float(atmos.aerosol_optical_depth),
        "ozone_over_300": float(atmos.total_ozone_du) / 300.0,
        "dominant_albedo": float(surface.dominant_albedo),
    }

    class_value = surface.surface_class.value
    for class_name in _KNOWN_CLASSES:
        feature_map[f"surface_class_is_{class_name}"] = 1.0 if class_value == class_name else 0.0

    landcover = dict(surface.landcover_mix)
    for class_name in _KNOWN_CLASSES:
        feature_map[f"landcover_mix_{class_name}"] = max(0.0, float(landcover.get(class_name, 0.0)))

    for field_name in _OPTIONAL_CLOUD_FIELDS:
        value = getattr(atmos, field_name)
        feature_map[field_name] = 0.0 if value is None else float(value)

    return feature_map


def featurize(
    dt: datetime,
    lat: float,
    lon: float,
    atmos: AtmosphereState,
    surface: SurfaceState,
    feature_names: list[str] | None = None,
) -> tuple[list[float], list[str]]:
    """Build deterministic feature vector in requested name order (missing -> 0.0)."""
    feature_map = compute_feature_dict(dt, lat, lon, atmos, surface)
    names = list(DEFAULT_FEATURE_NAMES if feature_names is None else feature_names)
    return [float(feature_map.get(name, 0.0)) for name in names], names
