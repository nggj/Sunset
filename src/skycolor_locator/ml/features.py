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


def featurize(
    dt: datetime,
    lat: float,
    lon: float,
    atmos: AtmosphereState,
    surface: SurfaceState,
) -> tuple[list[float], list[str]]:
    """Build fixed-order deterministic features for residual model inference."""
    dt_utc = _to_utc(dt)
    day_of_year = dt_utc.timetuple().tm_yday
    day_theta = 2.0 * pi * day_of_year / 365.0

    sza_deg, saz_deg, sun_elev_deg = solar_position(dt_utc, lat, lon)
    saz_rad = saz_deg * pi / 180.0

    turbidity = estimate_turbidity(atmos.aerosol_optical_depth, atmos.visibility_km)
    cod = atmos.cloud_optical_depth if atmos.cloud_optical_depth is not None else 0.0

    features: list[float] = [
        float(lat),
        abs(float(lat)),
        sin(day_theta),
        cos(day_theta),
        sun_elev_deg,
        sza_deg,
        sin(saz_rad),
        cos(saz_rad),
        float(turbidity),
        float(atmos.cloud_fraction),
        log1p(max(0.0, float(cod))),
        float(atmos.aerosol_optical_depth),
        float(atmos.total_ozone_du) / 300.0,
        float(surface.dominant_albedo),
    ]
    names: list[str] = [
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
    ]

    class_value = surface.surface_class.value
    for class_name in _KNOWN_CLASSES:
        names.append(f"surface_class_is_{class_name}")
        features.append(1.0 if class_value == class_name else 0.0)

    landcover = dict(surface.landcover_mix)
    for class_name in _KNOWN_CLASSES:
        names.append(f"landcover_mix_{class_name}")
        features.append(max(0.0, float(landcover.get(class_name, 0.0))))

    return features, names
