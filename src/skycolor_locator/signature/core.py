"""Core color-signature kernel for Skycolor Locator."""

from __future__ import annotations

from colorsys import rgb_to_hsv
from datetime import datetime
from math import floor
from typing import Any

from skycolor_locator.astro.solar import solar_position
from skycolor_locator.contracts import AtmosphereState, ColorSignature, SurfaceClass, SurfaceState
from skycolor_locator.ml.features import featurize
from skycolor_locator.ml.residual_model import ResidualHistogramModel
from skycolor_locator.sky.analytic import render_sky_rgb

# Spec-minimum quality-flag thresholds.
# - is_night: sun elevation <= 0° (below or on horizon)
# - low_sun: 0° < sun elevation < 10°
# - cloudy: cloud_fraction >= 0.6 or cloud_optical_depth >= 10
_LOW_SUN_ELEV_DEG = 10.0
_CLOUDY_FRACTION_THRESHOLD = 0.6
_CLOUD_OPTICAL_DEPTH_THRESHOLD = 10.0


def srgb_to_hsv(rgb: list[float]) -> tuple[float, float, float]:
    """Convert an sRGB pixel (0..1) to HSV."""
    if len(rgb) != 3:
        raise ValueError("rgb must contain 3 channels.")

    r = max(0.0, min(1.0, float(rgb[0])))
    g = max(0.0, min(1.0, float(rgb[1])))
    b = max(0.0, min(1.0, float(rgb[2])))
    return rgb_to_hsv(r, g, b)


def _flatten_rgb_grid(rgb: list[list[list[float]]]) -> list[list[float]]:
    """Flatten (H, W, 3) grid into a list of RGB triplets."""
    return [pixel for row in rgb for pixel in row]


def hue_histogram(
    rgb: list[list[list[float]]] | list[list[float]], bins: int, weight_mode: str = "sv"
) -> list[float]:
    """Compute normalized hue histogram from RGB pixels.

    Args:
        rgb: Either a `(H, W, 3)` nested list or flat `(N, 3)` list.
        bins: Number of hue bins.
        weight_mode: One of `"sv"`, `"s"`, or `"uniform"`.
    """
    if bins <= 0:
        raise ValueError("bins must be positive.")

    pixels: list[list[float]]
    if rgb and isinstance(rgb[0][0], list):
        pixels = _flatten_rgb_grid(rgb)  # type: ignore[arg-type]
    else:
        pixels = rgb  # type: ignore[assignment]

    hist = [0.0] * bins
    for pixel in pixels:
        h, s, v = srgb_to_hsv(pixel)
        idx = min(int(h * bins), bins - 1)
        if weight_mode == "sv":
            w = s * v
        elif weight_mode == "s":
            w = s
        elif weight_mode == "uniform":
            w = 1.0
        else:
            raise ValueError("weight_mode must be one of: sv, s, uniform")
        hist[idx] += w

    total = sum(hist)
    if total <= 0:
        return [1.0 / bins] * bins
    return [v / total for v in hist]


def smooth_circular(values: list[float], window: int = 3) -> list[float]:
    """Apply circular moving-average smoothing for periodic histograms."""
    if window <= 1:
        return values.copy()

    n = len(values)
    if n == 0:
        return []

    radius = window // 2
    out = [0.0] * n
    denom = float(2 * radius + 1)
    for i in range(n):
        acc = 0.0
        for k in range(-radius, radius + 1):
            acc += values[(i + k) % n]
        out[i] = acc / denom

    total = sum(out)
    if total > 0.0:
        out = [x / total for x in out]
    return out


def _surface_palette(surface: SurfaceState) -> dict[str, float]:
    """Build surface-class palette weights with normalized mixing."""
    mix = dict(surface.landcover_mix)
    if not mix:
        mix = {surface.surface_class.value: 1.0}

    total = sum(max(0.0, v) for v in mix.values())
    if total <= 0.0:
        return {surface.surface_class.value: 1.0}

    return {k: max(0.0, v) / total for k, v in mix.items()}


_DEFAULT_CLASS_RGB_PALETTE: dict[str, tuple[float, float, float]] = {
    SurfaceClass.OCEAN.value: (0.14, 0.38, 0.62),
    SurfaceClass.LAND.value: (0.45, 0.40, 0.26),
    SurfaceClass.URBAN.value: (0.50, 0.50, 0.52),
    SurfaceClass.SNOW.value: (0.92, 0.94, 0.98),
    SurfaceClass.DESERT.value: (0.80, 0.68, 0.42),
    SurfaceClass.FOREST.value: (0.12, 0.35, 0.16),
}


def _allocate_ground_sample_counts(palette: dict[str, float], total_samples: int) -> dict[str, int]:
    """Allocate a fixed number of samples across classes deterministically."""
    if total_samples <= 0:
        raise ValueError("ground_samples must be positive")

    items = sorted(palette.items(), key=lambda item: item[0])
    raw = [(name, weight * total_samples) for name, weight in items]

    counts = {name: int(floor(value)) for name, value in raw}
    remaining = total_samples - sum(counts.values())

    remainders = sorted(
        ((name, value - floor(value)) for name, value in raw),
        key=lambda item: (-item[1], item[0]),
    )
    for idx in range(remaining):
        name = remainders[idx % len(remainders)][0]
        counts[name] += 1

    return counts


def _compute_quality_flags(
    sun_elev_deg: float, atmos: AtmosphereState, turbidity: float
) -> list[str]:
    """Compute required+extra quality flags with deterministic ordering.

    Required flags from spec are emitted when conditions apply:
    - `is_night`: sun elevation <= 0°
    - `low_sun`: 0° < sun elevation < 10°
    - `missing_realtime`: provider reports missing realtime observations
    - `cloudy`: cloud fraction >= 0.6 OR cloud optical depth >= 10

    Existing extra flags are preserved (`high_cloud`, `high_turbidity`, `sun_below_horizon`, `ok`).
    """
    required: list[str] = []
    extra: list[str] = []

    cloud_fraction = max(0.0, min(1.0, atmos.cloud_fraction))
    cloud_optical_depth = atmos.cloud_optical_depth

    if sun_elev_deg <= 0.0:
        required.append("is_night")
    if 0.0 < sun_elev_deg < _LOW_SUN_ELEV_DEG:
        required.append("low_sun")
    if atmos.missing_realtime:
        required.append("missing_realtime")

    is_cloudy = cloud_fraction >= _CLOUDY_FRACTION_THRESHOLD or (
        cloud_optical_depth is not None and cloud_optical_depth >= _CLOUD_OPTICAL_DEPTH_THRESHOLD
    )
    if is_cloudy:
        required.append("cloudy")

    if cloud_fraction > 0.7:
        extra.append("high_cloud")
    if turbidity > 7.0:
        extra.append("high_turbidity")
    if sun_elev_deg < 0.0:
        extra.append("sun_below_horizon")

    combined = list(dict.fromkeys(required + extra))
    if not combined:
        combined.append("ok")
    return combined


def compute_color_signature(
    dt: datetime,
    lat: float,
    lon: float,
    atmos: AtmosphereState,
    surface: SurfaceState,
    config: dict[str, Any] | None = None,
) -> ColorSignature:
    """Compute sky/ground hue signature from analytic sky and simple ground model."""
    cfg = config or {}
    bins = int(cfg.get("bins", 36))
    n_az = int(cfg.get("n_az", 48))
    n_el = int(cfg.get("n_el", 24))
    smooth_window = int(cfg.get("smooth_window", 3))
    ground_samples = int(cfg.get("ground_samples", 2000))
    apply_residual = bool(cfg.get("apply_residual", False))
    residual_model_obj = cfg.get("residual_model")
    residual_model = residual_model_obj if isinstance(residual_model_obj, ResidualHistogramModel) else None

    sky_rgb, sky_meta = render_sky_rgb(dt=dt, lat=lat, lon=lon, atmos=atmos, n_az=n_az, n_el=n_el)
    sky_hist = hue_histogram(sky_rgb, bins=bins, weight_mode="sv")
    sky_hist = smooth_circular(sky_hist, window=smooth_window)

    horizon = sky_rgb[0]
    horizon_color = [sum(pixel[c] for pixel in horizon) / len(horizon) for c in range(3)]

    palette = _surface_palette(surface)
    albedo = max(0.0, min(1.0, surface.dominant_albedo))
    _, _, sun_elev_deg = solar_position(dt, lat, lon)
    illum = max(0.15, min(1.0, 0.2 + max(sun_elev_deg, 0.0) / 90.0))
    haze = max(
        0.0, min(0.85, 0.15 + 0.55 * atmos.cloud_fraction + 0.2 * atmos.aerosol_optical_depth)
    )

    class_counts = _allocate_ground_sample_counts(palette, ground_samples)

    ground_pixels: list[list[float]] = []
    default_land = _DEFAULT_CLASS_RGB_PALETTE[SurfaceClass.LAND.value]
    for name in sorted(palette):
        override = surface.class_rgb.get(name)
        base = override if override is not None else _DEFAULT_CLASS_RGB_PALETTE.get(name, default_land)
        lit = [base[i] * (0.35 + 0.65 * illum) * (0.5 + 0.5 * albedo) for i in range(3)]
        mixed = [lit[i] * (1.0 - haze) + horizon_color[i] * haze for i in range(3)]
        ground_pixels.extend([mixed] * class_counts[name])

    ground_hist = hue_histogram(ground_pixels, bins=bins, weight_mode="sv")
    ground_hist = smooth_circular(ground_hist, window=smooth_window)

    signature = sky_hist + ground_hist
    hue_bins = [i / bins for i in range(bins)]

    cloud = max(0.0, min(1.0, atmos.cloud_fraction))
    turbidity = float(sky_meta.get("turbidity", 3.0))
    uncertainty_score = max(0.0, min(1.0, 0.15 + 0.45 * cloud + 0.05 * turbidity / 10.0))
    quality_flags = _compute_quality_flags(
        sun_elev_deg=sun_elev_deg, atmos=atmos, turbidity=turbidity
    )

    meta: dict[str, Any] = {
        "sun_elev_deg": sun_elev_deg,
        "sza_deg": sky_meta.get("sza_deg"),
        "saz_deg": sky_meta.get("saz_deg"),
        "turbidity": turbidity,
        "quality_flags": quality_flags,
        "uncertainty_score": uncertainty_score,
        "ground_sample_count": len(ground_pixels),
    }

    baseline = ColorSignature(
        hue_bins=hue_bins,
        sky_hue_hist=sky_hist,
        ground_hue_hist=ground_hist,
        signature=signature,
        meta=meta,
        uncertainty_score=uncertainty_score,
        quality_flags=quality_flags,
    )

    if apply_residual:
        if residual_model is None:
            raise ValueError("apply_residual=True requires residual_model in config")
        features, _ = featurize(dt=dt, lat=lat, lon=lon, atmos=atmos, surface=surface)
        return residual_model.apply_to_signature(baseline, features)

    return baseline
