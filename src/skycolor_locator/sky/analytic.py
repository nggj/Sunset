"""Analytic sky dome renderer using a Preetham/Perez xyY pipeline.

This module keeps a deterministic pure-Python implementation and does not require
NumPy. If later optimization is needed, vectorized paths can be added while
preserving the same output contract.
"""

from __future__ import annotations

from datetime import datetime
from math import acos, cos, exp, pi, sin, tan
from typing import Any

from skycolor_locator.astro.solar import solar_position
from skycolor_locator.contracts import AtmosphereState


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value into an inclusive range."""
    return max(lower, min(upper, value))


def estimate_turbidity(aod: float, visibility_km: float | None) -> float:
    """Estimate turbidity from AOD and optional visibility heuristic."""
    if visibility_km is None:
        vis_term = 0.0
    else:
        vis_term = _clamp((20.0 - visibility_km) / 20.0, 0.0, 1.0)

    turbidity = 2.0 + 8.0 * _clamp(aod, 0.0, 1.0) + 2.5 * vis_term
    return _clamp(turbidity, 1.8, 10.0)


def _perez_coeffs(turbidity: float) -> dict[str, tuple[float, float, float, float, float]]:
    """Return Preetham/Perez coefficients for Y, x, y distributions."""
    return {
        "Y": (
            0.1787 * turbidity - 1.4630,
            -0.3554 * turbidity + 0.4275,
            -0.0227 * turbidity + 5.3251,
            0.1206 * turbidity - 2.5771,
            -0.0670 * turbidity + 0.3703,
        ),
        "x": (
            -0.0193 * turbidity - 0.2592,
            -0.0665 * turbidity + 0.0008,
            -0.0004 * turbidity + 0.2125,
            -0.0641 * turbidity - 0.8989,
            -0.0033 * turbidity + 0.0452,
        ),
        "y": (
            -0.0167 * turbidity - 0.2608,
            -0.0950 * turbidity + 0.0092,
            -0.0079 * turbidity + 0.2102,
            -0.0441 * turbidity - 1.6537,
            -0.0109 * turbidity + 0.0529,
        ),
    }


def _perez_distribution(theta: float, gamma: float, coeffs: tuple[float, float, float, float, float]) -> float:
    """Evaluate Perez distribution function F(theta, gamma)."""
    a, b, c, d, e = coeffs
    cos_theta = max(cos(theta), 0.01)
    return (1.0 + a * exp(b / cos_theta)) * (1.0 + c * exp(d * gamma) + e * cos(gamma) ** 2)


def _zenith_luminance_yz(turbidity: float, theta_s: float) -> float:
    """Compute zenith luminance Yz from Preetham model."""
    chi = (4.0 / 9.0 - turbidity / 120.0) * (pi - 2.0 * theta_s)
    return (4.0453 * turbidity - 4.9710) * tan(chi) - 0.2155 * turbidity + 2.4192


def _zenith_chromaticity_xy(turbidity: float, theta_s: float) -> tuple[float, float]:
    """Compute zenith chromaticity (xz, yz) via polynomial approximation."""
    t2 = turbidity * turbidity
    th = theta_s
    th2 = th * th
    th3 = th2 * th

    xz = (
        (0.00166 * th3 - 0.00375 * th2 + 0.00209 * th + 0.0) * t2
        + (-0.02903 * th3 + 0.06377 * th2 - 0.03202 * th + 0.00394) * turbidity
        + (0.11693 * th3 - 0.21196 * th2 + 0.06052 * th + 0.25886)
    )
    yz = (
        (0.00275 * th3 - 0.00610 * th2 + 0.00317 * th + 0.0) * t2
        + (-0.04214 * th3 + 0.08970 * th2 - 0.04153 * th + 0.00516) * turbidity
        + (0.15346 * th3 - 0.26756 * th2 + 0.06670 * th + 0.26688)
    )

    return xz, yz


def _xyy_to_xyz(x: float, y: float, y_lum: float) -> tuple[float, float, float]:
    """Convert xyY color to XYZ."""
    y_safe = max(y, 1e-6)
    x_val = x * (y_lum / y_safe)
    z_val = (1.0 - x - y) * (y_lum / y_safe)
    return x_val, y_lum, z_val


def _xyz_to_srgb(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert XYZ to gamma-corrected sRGB in [0, 1]."""
    r_lin = 3.2406 * x - 1.5372 * y - 0.4986 * z
    g_lin = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b_lin = 0.0557 * x - 0.2040 * y + 1.0570 * z

    def gamma_encode(v: float) -> float:
        v_clamped = max(v, 0.0)
        if v_clamped <= 0.0031308:
            return 12.92 * v_clamped
        return float(1.055 * (v_clamped ** (1.0 / 2.4)) - 0.055)

    return (
        _clamp(gamma_encode(r_lin), 0.0, 1.0),
        _clamp(gamma_encode(g_lin), 0.0, 1.0),
        _clamp(gamma_encode(b_lin), 0.0, 1.0),
    )


def _ozone_rgb_correction(rgb: tuple[float, float, float], atmos: AtmosphereState, sza_deg: float) -> tuple[float, float, float]:
    """Apply a simple ozone-dependent color correction heuristic."""
    sza_term = _clamp((sza_deg - 50.0) / 40.0, 0.0, 1.0)
    k = 0.25 * (atmos.total_ozone_du / 300.0) * sza_term * exp(-2.0 * max(atmos.aerosol_optical_depth, 0.0))
    corr = (1.0 - 0.35 * k, 1.0 - 0.55 * k, 1.0)
    return (
        _clamp(rgb[0] * corr[0], 0.0, 1.0),
        _clamp(rgb[1] * corr[1], 0.0, 1.0),
        _clamp(rgb[2] * corr[2], 0.0, 1.0),
    )


def _apply_cloud_blend(rgb: tuple[float, float, float], atmos: AtmosphereState) -> tuple[float, float, float]:
    """Blend clear-sky RGB toward overcast colors."""
    cloud_fraction = _clamp(atmos.cloud_fraction, 0.0, 1.0)
    cod = 0.0 if atmos.cloud_optical_depth is None else max(atmos.cloud_optical_depth, 0.0)
    blend = _clamp(cloud_fraction * (1.0 - exp(-0.3 * cod if cod > 0.0 else -0.3 * 3.0)), 0.0, 1.0)
    overcast = (0.75, 0.75, 0.78)
    return (
        _clamp((1.0 - blend) * rgb[0] + blend * overcast[0], 0.0, 1.0),
        _clamp((1.0 - blend) * rgb[1] + blend * overcast[1], 0.0, 1.0),
        _clamp((1.0 - blend) * rgb[2] + blend * overcast[2], 0.0, 1.0),
    )


def _gaussian_kernel1d(sigma: float) -> list[float]:
    """Build normalized 1D Gaussian kernel."""
    if sigma <= 0.0:
        return [1.0]
    radius = max(1, int(3.0 * sigma))
    kernel = [exp(-(x * x) / (2.0 * sigma * sigma)) for x in range(-radius, radius + 1)]
    total = sum(kernel)
    return [k / total for k in kernel]


def _gaussian_blur_rgb(rgb: list[list[list[float]]], sigma: float) -> list[list[list[float]]]:
    """Apply lightweight separable Gaussian blur on sky RGB grid."""
    if sigma <= 0.0:
        return rgb

    kernel = _gaussian_kernel1d(sigma)
    radius = len(kernel) // 2
    n_el = len(rgb)
    n_az = len(rgb[0]) if n_el else 0

    temp = [[[0.0, 0.0, 0.0] for _ in range(n_az)] for _ in range(n_el)]
    out = [[[0.0, 0.0, 0.0] for _ in range(n_az)] for _ in range(n_el)]

    # Azimuth blur with circular wrapping.
    for i in range(n_el):
        for j in range(n_az):
            for k_idx, w in enumerate(kernel):
                dj = k_idx - radius
                src_j = (j + dj) % n_az
                temp[i][j][0] += w * rgb[i][src_j][0]
                temp[i][j][1] += w * rgb[i][src_j][1]
                temp[i][j][2] += w * rgb[i][src_j][2]

    # Elevation blur with edge clamping.
    for i in range(n_el):
        for j in range(n_az):
            for k_idx, w in enumerate(kernel):
                di = k_idx - radius
                src_i = _clamp(i + di, 0, n_el - 1)
                src_i_int = int(src_i)
                out[i][j][0] += w * temp[src_i_int][j][0]
                out[i][j][1] += w * temp[src_i_int][j][1]
                out[i][j][2] += w * temp[src_i_int][j][2]

            out[i][j][0] = _clamp(out[i][j][0], 0.0, 1.0)
            out[i][j][1] = _clamp(out[i][j][1], 0.0, 1.0)
            out[i][j][2] = _clamp(out[i][j][2], 0.0, 1.0)

    return out


def render_sky_rgb(
    dt: datetime,
    lat: float,
    lon: float,
    atmos: AtmosphereState,
    n_az: int,
    n_el: int,
) -> tuple[list[list[list[float]]], dict[str, Any]]:
    """Render a sky dome RGB grid using a Preetham/Perez xyY pipeline.

    Returns:
        `(rgb, meta)` where `rgb` shape is `(n_el, n_az, 3)` represented as
        nested Python lists with values clipped to `[0, 1]`.
    """
    if n_az <= 0 or n_el <= 0:
        raise ValueError("n_az and n_el must be positive.")

    sza_deg, saz_deg, sun_elev_deg = solar_position(dt, lat, lon)
    theta_s = _clamp(sza_deg * pi / 180.0, 0.0, pi / 2.0)
    phi_s = saz_deg * pi / 180.0

    turbidity = estimate_turbidity(atmos.aerosol_optical_depth, atmos.visibility_km)
    coeffs = _perez_coeffs(turbidity)

    yz = _zenith_luminance_yz(turbidity, theta_s)
    xz, yy = _zenith_chromaticity_xy(turbidity, theta_s)

    fy0 = _perez_distribution(0.0, theta_s, coeffs["Y"])
    fx0 = _perez_distribution(0.0, theta_s, coeffs["x"])
    fyy0 = _perez_distribution(0.0, theta_s, coeffs["y"])

    rgb_grid: list[list[list[float]]] = []
    for el_idx in range(n_el):
        elev = (el_idx / max(n_el - 1, 1)) * (pi / 2.0)
        theta = pi / 2.0 - elev
        row: list[list[float]] = []

        for az_idx in range(n_az):
            phi = (az_idx / n_az) * 2.0 * pi

            cos_gamma = _clamp(
                sin(theta) * sin(theta_s) * cos(phi - phi_s) + cos(theta) * cos(theta_s),
                -1.0,
                1.0,
            )
            gamma = acos(cos_gamma)

            fy = _perez_distribution(theta, gamma, coeffs["Y"])
            fx = _perez_distribution(theta, gamma, coeffs["x"])
            fyy = _perez_distribution(theta, gamma, coeffs["y"])

            y_lum = yz * fy / max(fy0, 1e-6)
            x_chroma = xz * fx / max(fx0, 1e-6)
            y_chroma = yy * fyy / max(fyy0, 1e-6)

            # Normalize luminance for histogram-oriented rendering.
            y_norm = y_lum / max(yz, 1e-6)
            x_chroma = _clamp(x_chroma, 0.001, 0.999)
            y_chroma = _clamp(y_chroma, 0.001, 0.999)
            if x_chroma + y_chroma > 0.999:
                scale = 0.999 / (x_chroma + y_chroma)
                x_chroma *= scale
                y_chroma *= scale

            x_val, y_val, z_val = _xyy_to_xyz(x_chroma, y_chroma, y_norm)
            rgb = _xyz_to_srgb(x_val, y_val, z_val)
            rgb = _ozone_rgb_correction(rgb, atmos, sza_deg)
            rgb = _apply_cloud_blend(rgb, atmos)

            row.append([rgb[0], rgb[1], rgb[2]])
        rgb_grid.append(row)

    # Optional lightweight post smoothing (off by default).
    apply_smoothing = bool(getattr(atmos, "apply_smoothing", False))
    if apply_smoothing:
        sigma = float(getattr(atmos, "smoothing_sigma", 0.8))
        rgb_grid = _gaussian_blur_rgb(rgb_grid, sigma=sigma)

    meta: dict[str, Any] = {
        "sun_elev_deg": sun_elev_deg,
        "sza_deg": sza_deg,
        "saz_deg": saz_deg,
        "turbidity": turbidity,
    }
    return rgb_grid, meta
