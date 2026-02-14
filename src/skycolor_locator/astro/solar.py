"""Solar position helpers.

This module provides a deterministic approximation suitable for MVP tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import acos, atan2, cos, degrees, radians, sin


def _normalize_degrees(angle_deg: float) -> float:
    """Normalize an angle to [0, 360)."""
    return angle_deg % 360.0


def solar_position(dt: datetime, lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """Compute solar zenith/azimuth/elevation for UTC datetime and WGS84 coordinates.

    Args:
        dt: Time as a timezone-aware datetime. Converted to UTC internally.
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees (east positive).

    Returns:
        Tuple of `(sza_deg, saz_deg, sun_elev_deg)` where:
        - `sza_deg`: solar zenith angle
        - `saz_deg`: solar azimuth angle, normalized to [0, 360)
        - `sun_elev_deg`: solar elevation angle
    """
    if dt.tzinfo is None:
        raise ValueError("dt must be timezone-aware.")

    dt_utc = dt.astimezone(timezone.utc)
    day_of_year = dt_utc.timetuple().tm_yday
    hour_fraction = (
        dt_utc.hour
        + dt_utc.minute / 60.0
        + dt_utc.second / 3600.0
        + dt_utc.microsecond / 3_600_000_000.0
    )

    gamma = 2.0 * 3.141592653589793 * (day_of_year - 1 + (hour_fraction - 12.0) / 24.0) / 365.0

    decl_rad = (
        0.006918
        - 0.399912 * cos(gamma)
        + 0.070257 * sin(gamma)
        - 0.006758 * cos(2.0 * gamma)
        + 0.000907 * sin(2.0 * gamma)
        - 0.002697 * cos(3.0 * gamma)
        + 0.00148 * sin(3.0 * gamma)
    )

    eqtime_min = 229.18 * (
        0.000075
        + 0.001868 * cos(gamma)
        - 0.032077 * sin(gamma)
        - 0.014615 * cos(2.0 * gamma)
        - 0.040849 * sin(2.0 * gamma)
    )

    true_solar_time_min = (hour_fraction * 60.0 + eqtime_min + 4.0 * lon_deg) % 1440.0
    hour_angle_deg = true_solar_time_min / 4.0 - 180.0
    hour_angle_rad = radians(hour_angle_deg)

    lat_rad = radians(lat_deg)
    cos_zenith = sin(lat_rad) * sin(decl_rad) + cos(lat_rad) * cos(decl_rad) * cos(hour_angle_rad)
    cos_zenith = min(1.0, max(-1.0, cos_zenith))

    sza_deg = degrees(acos(cos_zenith))
    sun_elev_deg = 90.0 - sza_deg

    azimuth_rad = atan2(
        sin(hour_angle_rad),
        cos(hour_angle_rad) * sin(lat_rad) - tan_safe(decl_rad) * cos(lat_rad),
    )
    saz_deg = _normalize_degrees(degrees(azimuth_rad) + 180.0)

    return (sza_deg, saz_deg, sun_elev_deg)


def tan_safe(value_rad: float) -> float:
    """Return tangent using sine/cosine ratio for numerical stability."""
    cos_v = cos(value_rad)
    if abs(cos_v) < 1e-12:
        return sin(value_rad) / 1e-12
    return sin(value_rad) / cos_v
