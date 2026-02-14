"""Analytic sky dome baseline renderer.

Implements a lightweight Perez-style sky model with deterministic heuristics for
cloud and ozone corrections.
"""

from __future__ import annotations

from datetime import datetime
from math import acos, cos, exp, pi, sin
from typing import Any

from skycolor_locator.astro.solar import solar_position
from skycolor_locator.contracts import AtmosphereState


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value into an inclusive range."""
    return max(lower, min(upper, value))


def estimate_turbidity(aod: float, visibility_km: float | None) -> float:
    """Estimate turbidity from AOD and optional visibility.

    Lower visibility and higher AOD imply larger turbidity.
    """
    vis_term = 0.0
    if visibility_km is not None:
        vis_term = _clamp((20.0 - visibility_km) / 20.0, 0.0, 1.0)

    turbidity = 2.0 + 8.0 * _clamp(aod, 0.0, 1.0) + 2.5 * vis_term
    return _clamp(turbidity, 1.8, 10.0)


def _perez_coefficients_y(turbidity: float) -> tuple[float, float, float, float, float]:
    """Return Perez coefficients for relative luminance distribution."""
    return (
        0.1787 * turbidity - 1.4630,
        -0.3554 * turbidity + 0.4275,
        -0.0227 * turbidity + 5.3251,
        0.1206 * turbidity - 2.5771,
        -0.0670 * turbidity + 0.3703,
    )


def _perez_distribution(theta: float, gamma: float, coeffs: tuple[float, float, float, float, float]) -> float:
    """Evaluate Perez sky distribution term."""
    a, b, c, d, e = coeffs
    cos_theta = max(cos(theta), 0.01)
    return (1.0 + a * exp(b / cos_theta)) * (1.0 + c * exp(d * gamma) + e * cos(gamma) ** 2)


def _mix_rgb(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    """Linearly interpolate between two RGB colors."""
    return (
        a[0] * (1.0 - t) + b[0] * t,
        a[1] * (1.0 - t) + b[1] * t,
        a[2] * (1.0 - t) + b[2] * t,
    )


def render_sky_rgb(
    dt: datetime,
    lat: float,
    lon: float,
    atmos: AtmosphereState,
    n_az: int,
    n_el: int,
) -> tuple[list[list[list[float]]], dict[str, Any]]:
    """Render a sky dome RGB grid using a Perez-based analytic baseline.

    Returns:
        `(rgb, meta)` where `rgb` shape is `(n_el, n_az, 3)` represented as
        nested Python lists with values clipped to `[0, 1]`.
    """
    if n_az <= 0 or n_el <= 0:
        raise ValueError("n_az and n_el must be positive.")

    sza_deg, saz_deg, sun_elev_deg = solar_position(dt, lat, lon)
    sun_theta = _clamp(sza_deg * pi / 180.0, 0.0, pi / 2)
    sun_az = saz_deg * pi / 180.0

    turbidity = estimate_turbidity(atmos.aerosol_optical_depth, atmos.visibility_km)
    coeffs = _perez_coefficients_y(turbidity)

    # Approximate zenith and horizon colors, shifted by turbidity and ozone.
    turb_norm = (turbidity - 1.8) / (10.0 - 1.8)
    ozone_norm = _clamp((atmos.total_ozone_du - 250.0) / 200.0, 0.0, 1.0)
    cloud = _clamp(atmos.cloud_fraction, 0.0, 1.0)

    zenith_blue = (0.16 + 0.08 * turb_norm, 0.38 + 0.10 * turb_norm, 0.95 - 0.20 * turb_norm)
    horizon_warm = (0.85 - 0.15 * ozone_norm, 0.62 - 0.10 * ozone_norm, 0.38 - 0.05 * ozone_norm)

    y_sun = _perez_distribution(sun_theta, 0.0, coeffs)
    sun_height_gain = _clamp(0.2 + max(0.0, sin(sun_elev_deg * pi / 180.0)), 0.2, 1.2)

    rgb_grid: list[list[list[float]]] = []
    for el_idx in range(n_el):
        elev = (el_idx / max(n_el - 1, 1)) * (pi / 2.0)
        theta = pi / 2.0 - elev
        row: list[list[float]] = []

        horizon_mix = _clamp((1.0 - sin(elev)) ** 0.7, 0.0, 1.0)
        base_color = _mix_rgb(zenith_blue, horizon_warm, horizon_mix)

        for az_idx in range(n_az):
            az = (az_idx / n_az) * 2.0 * pi
            cos_gamma = _clamp(
                sin(elev) * sin(pi / 2.0 - sun_theta)
                + cos(elev) * cos(pi / 2.0 - sun_theta) * cos(az - sun_az),
                -1.0,
                1.0,
            )
            gamma = acos(cos_gamma)

            rel_lum = _perez_distribution(theta, gamma, coeffs) / max(y_sun, 1e-6)
            lum = _clamp((0.25 + 0.75 * rel_lum) * sun_height_gain, 0.0, 1.0)

            r = base_color[0] * lum
            g = base_color[1] * lum
            b = base_color[2] * lum

            # Ozone absorbs more in shorter wavelengths (heuristic blue attenuation).
            ozone_absorb = 1.0 - 0.12 * ozone_norm
            b *= ozone_absorb

            # Cloud correction: blend toward gray + brightness damping.
            gray = (r + g + b) / 3.0
            desat = 0.75 * cloud
            bright_damp = 1.0 - 0.35 * cloud

            r = ((1.0 - desat) * r + desat * gray) * bright_damp
            g = ((1.0 - desat) * g + desat * gray) * bright_damp
            b = ((1.0 - desat) * b + desat * gray) * bright_damp

            row.append([_clamp(r, 0.0, 1.0), _clamp(g, 0.0, 1.0), _clamp(b, 0.0, 1.0)])
        rgb_grid.append(row)

    meta: dict[str, Any] = {
        "sun_elev_deg": sun_elev_deg,
        "sza_deg": sza_deg,
        "saz_deg": saz_deg,
        "turbidity": turbidity,
        "cloud_fraction": cloud,
        "ozone_norm": ozone_norm,
    }
    return rgb_grid, meta
