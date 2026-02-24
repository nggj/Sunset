"""Perceptual search embedding for sky/ground color conditions."""

from __future__ import annotations

from colorsys import rgb_to_hsv
from datetime import datetime
from math import floor, sqrt
from typing import Any

from skycolor_locator.astro.solar import solar_position
from skycolor_locator.contracts import AtmosphereState, CameraProfile, SurfaceClass, SurfaceState
from skycolor_locator.sky.analytic import render_sky_rgb
from skycolor_locator.view.fov import sample_sky_in_camera_view
from skycolor_locator.view.horizon import FlatHorizonModel, HorizonModel

H_BINS = 36
S_BINS = 8
V_BINS = 8
PROFILE_BINS = 16
SKY_BANDS = 3


def srgb_to_hsv(pixel: list[float]) -> tuple[float, float, float]:
    """Convert one sRGB pixel in [0, 1] to HSV."""
    if len(pixel) != 3:
        raise ValueError("pixel must contain 3 channels")
    r = max(0.0, min(1.0, float(pixel[0])))
    g = max(0.0, min(1.0, float(pixel[1])))
    b = max(0.0, min(1.0, float(pixel[2])))
    return rgb_to_hsv(r, g, b)


def flatten_sky_pixels(sky_rgb: list[list[list[float]]]) -> list[list[float]]:
    """Flatten (n_el, n_az, 3) sky grid into a flat pixel list."""
    return [pixel for row in sky_rgb for pixel in row]


def luma(pixel: list[float]) -> float:
    """Compute Rec.709 luma from one RGB pixel."""
    return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]


def histogram_1d(
    values: list[float],
    bins: int,
    value_range: tuple[float, float],
    weights: list[float] | None = None,
) -> list[float]:
    """Build normalized 1D histogram in pure Python."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    if len(values) == 0:
        return [1.0 / bins] * bins
    if weights is not None and len(weights) != len(values):
        raise ValueError("weights length must match values length")

    low, high = value_range
    if high <= low:
        raise ValueError("value_range must satisfy high > low")

    hist = [0.0] * bins
    scale = bins / (high - low)
    for idx, value in enumerate(values):
        clipped = min(max(value, low), high)
        bin_index = min(int((clipped - low) * scale), bins - 1)
        w = 1.0 if weights is None else max(0.0, weights[idx])
        hist[bin_index] += w

    total = sum(hist)
    if total <= 0.0:
        return [1.0 / bins] * bins
    return [v / total for v in hist]


def hue_histogram_from_pixels(
    pixels: list[list[float]], bins: int, weight_mode: str = "sv"
) -> list[float]:
    """Compute normalized hue histogram from flat RGB pixels."""
    hues: list[float] = []
    weights: list[float] = []
    for pixel in pixels:
        h, s, v = srgb_to_hsv(pixel)
        hues.append(h)
        if weight_mode == "sv":
            weights.append(s * v)
        elif weight_mode == "s":
            weights.append(s)
        elif weight_mode == "uniform":
            weights.append(1.0)
        else:
            raise ValueError("weight_mode must be one of: sv, s, uniform")
    return histogram_1d(hues, bins=bins, value_range=(0.0, 1.0), weights=weights)


def l2_normalize(vec: list[float]) -> list[float]:
    """Return L2-normalized copy of the input vector."""
    norm = sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return vec.copy()
    return [v / norm for v in vec]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return sqrt(sum((v - mu) * (v - mu) for v in values) / len(values))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    q_clamped = min(1.0, max(0.0, q))
    idx = int(round(q_clamped * (len(ordered) - 1)))
    return ordered[idx]


def colorfulness_metric(pixels: list[list[float]]) -> float:
    """Compute Hasler-Susstrunk colorfulness metric from RGB pixels."""
    if not pixels:
        return 0.0

    rg: list[float] = []
    yb: list[float] = []
    for r, g, b in pixels:
        rg.append(r - g)
        yb.append(0.5 * (r + g) - b)

    std_rg = _std(rg)
    std_yb = _std(yb)
    mean_rg = _mean(rg)
    mean_yb = _mean(yb)

    return sqrt(std_rg * std_rg + std_yb * std_yb) + 0.3 * sqrt(
        mean_rg * mean_rg + mean_yb * mean_yb
    )


def downsample_profile(profile: list[float], out_bins: int) -> list[float]:
    """Downsample profile to fixed bins with deterministic averaging buckets."""
    if out_bins <= 0:
        raise ValueError("out_bins must be positive")
    if not profile:
        return [0.0] * out_bins

    n = len(profile)
    out: list[float] = []
    for i in range(out_bins):
        start = int(floor(i * n / out_bins))
        end = int(floor((i + 1) * n / out_bins))
        if end <= start:
            end = min(start + 1, n)
        bucket = profile[start:end]
        out.append(_mean(bucket))
    return out


def _surface_palette(surface: SurfaceState) -> dict[str, float]:
    """Build normalized deterministic surface class weights."""
    mix = dict(surface.landcover_mix)
    if not mix:
        mix = {surface.surface_class.value: 1.0}

    total = sum(max(0.0, value) for value in mix.values())
    if total <= 0.0:
        return {surface.surface_class.value: 1.0}
    return {name: max(0.0, value) / total for name, value in mix.items()}


def _class_to_rgb(name: str) -> tuple[float, float, float]:
    """Map landcover class to representative color."""
    palette = {
        SurfaceClass.OCEAN.value: (0.14, 0.38, 0.62),
        SurfaceClass.LAND.value: (0.45, 0.40, 0.26),
        SurfaceClass.URBAN.value: (0.50, 0.50, 0.52),
        SurfaceClass.SNOW.value: (0.92, 0.94, 0.98),
        SurfaceClass.DESERT.value: (0.80, 0.68, 0.42),
        SurfaceClass.FOREST.value: (0.12, 0.35, 0.16),
    }
    return palette.get(name, (0.45, 0.40, 0.26))


def _allocate_ground_sample_counts(palette: dict[str, float], total_samples: int) -> dict[str, int]:
    """Allocate deterministic integer ground samples by class weights."""
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
    for i in range(remaining):
        counts[remainders[i % len(remainders)][0]] += 1
    return counts


def _build_ground_pixels(
    dt: datetime,
    lat: float,
    lon: float,
    atmos: AtmosphereState,
    surface: SurfaceState,
    horizon_color: list[float],
    ground_samples: int,
) -> list[list[float]]:
    """Build deterministic synthetic ground pixel population."""
    palette = _surface_palette(surface)
    albedo = max(0.0, min(1.0, surface.dominant_albedo))
    _, _, sun_elev_deg = solar_position(dt, lat, lon)
    illum = max(0.15, min(1.0, 0.2 + max(sun_elev_deg, 0.0) / 90.0))
    haze = max(
        0.0,
        min(0.85, 0.15 + 0.55 * atmos.cloud_fraction + 0.2 * atmos.aerosol_optical_depth),
    )

    class_counts = _allocate_ground_sample_counts(palette, ground_samples)

    ground_pixels: list[list[float]] = []
    for name in sorted(palette):
        base = _class_to_rgb(name)
        lit = [base[i] * (0.35 + 0.65 * illum) * (0.5 + 0.5 * albedo) for i in range(3)]
        mixed = [lit[i] * (1.0 - haze) + horizon_color[i] * haze for i in range(3)]
        ground_pixels.extend([mixed] * class_counts[name])
    return ground_pixels


def compute_perceptual_v1_from_buffers(
    sky_rgb: list[list[list[float]]],
    ground_pixels: list[list[float]],
    view_grid: list[list[list[float] | None]] | None = None,
    no_ground: bool = False,
) -> tuple[list[float], dict[str, Any]]:
    """Compute perceptual_v1 feature vector from precomputed sky/ground buffers."""
    if not sky_rgb or not sky_rgb[0]:
        raise ValueError("sky_rgb must be non-empty")

    n_el = len(sky_rgb)
    band_features: list[float] = []
    band_sizes: list[int] = []
    for band_index in range(SKY_BANDS):
        start = int(floor(band_index * n_el / SKY_BANDS))
        end = int(floor((band_index + 1) * n_el / SKY_BANDS))
        if end <= start:
            end = min(start + 1, n_el)
        band_pixels = [pixel for row in sky_rgb[start:end] for pixel in row]
        band_sizes.append(len(band_pixels))
        band_features.extend(hue_histogram_from_pixels(band_pixels, bins=H_BINS, weight_mode="sv"))

    if view_grid is None:
        sky_pixels = flatten_sky_pixels(sky_rgb)
        row_luma = [_mean([luma(pixel) for pixel in row]) for row in sky_rgb]
    else:
        sky_pixels = [pixel for row in view_grid for pixel in row if pixel is not None]
        row_luma = [
            _mean([luma(pixel) for pixel in row if pixel is not None])
            for row in view_grid
            if any(pixel is not None for pixel in row)
        ]

    sky_hsv = [srgb_to_hsv(pixel) for pixel in sky_pixels]
    sky_sat = [s for _, s, _ in sky_hsv]
    sky_val = [v for _, _, v in sky_hsv]

    sky_sat_hist = histogram_1d(sky_sat, bins=S_BINS, value_range=(0.0, 1.0), weights=sky_val)
    sky_val_hist = histogram_1d(sky_val, bins=V_BINS, value_range=(0.0, 1.0))

    sky_profile = downsample_profile(row_luma, out_bins=PROFILE_BINS)

    sky_lumas = [luma(pixel) for pixel in sky_pixels]
    sky_luma_std = _std(sky_lumas)
    sky_luma_p90_minus_p10 = _percentile(sky_lumas, 0.9) - _percentile(sky_lumas, 0.1)
    sky_colorfulness = colorfulness_metric(sky_pixels)

    if no_ground:
        ground_hue_hist = [1.0 / H_BINS] * H_BINS
        ground_sat_hist = [1.0 / S_BINS] * S_BINS
        ground_val_hist = [1.0 / V_BINS] * V_BINS
        ground_luma_mean = 0.0
        ground_luma_std = 0.0
        ground_colorfulness = 0.0
    else:
        ground_hsv = [srgb_to_hsv(pixel) for pixel in ground_pixels]
        ground_hues = [h for h, _, _ in ground_hsv]
        ground_sat = [s for _, s, _ in ground_hsv]
        ground_val = [v for _, _, v in ground_hsv]

        ground_hue_hist = histogram_1d(
            ground_hues,
            bins=H_BINS,
            value_range=(0.0, 1.0),
            weights=[s * v for _, s, v in ground_hsv],
        )
        ground_sat_hist = histogram_1d(
            ground_sat, bins=S_BINS, value_range=(0.0, 1.0), weights=ground_val
        )
        ground_val_hist = histogram_1d(ground_val, bins=V_BINS, value_range=(0.0, 1.0))

        ground_lumas = [luma(pixel) for pixel in ground_pixels]
        ground_luma_mean = _mean(ground_lumas)
        ground_luma_std = _std(ground_lumas)
        ground_colorfulness = colorfulness_metric(ground_pixels)

    raw = (
        band_features
        + sky_sat_hist
        + sky_val_hist
        + sky_profile
        + [sky_luma_std, sky_luma_p90_minus_p10, sky_colorfulness]
        + ground_hue_hist
        + ground_sat_hist
        + ground_val_hist
        + [ground_luma_mean, ground_luma_std, ground_colorfulness]
    )
    vector = l2_normalize(raw)

    meta: dict[str, Any] = {
        "vector_type": "perceptual_v1",
        "vector_dim": len(vector),
        "components": {
            "sky_band_sizes": band_sizes,
            "sky_rows": n_el,
            "ground_pixel_count": len(ground_pixels),
        },
        "no_ground": no_ground,
    }
    return vector, meta


def compute_perceptual_v1(
    dt: datetime,
    lat: float,
    lon: float,
    atmos: AtmosphereState,
    surface: SurfaceState,
    config: dict[str, Any] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Compute perceptual_v1 vector using analytic sky and deterministic ground model."""
    cfg = config or {}
    n_az = int(cfg.get("n_az", 48))
    n_el = int(cfg.get("n_el", 24))
    ground_samples = int(cfg.get("ground_samples", 2000))

    sky_rgb, sky_meta = render_sky_rgb(dt=dt, lat=lat, lon=lon, atmos=atmos, n_az=n_az, n_el=n_el)

    camera_cfg = cfg.get("camera_profile")
    camera_profile: CameraProfile | None
    if isinstance(camera_cfg, CameraProfile):
        camera_profile = camera_cfg
    elif isinstance(camera_cfg, dict):
        camera_profile = CameraProfile(**camera_cfg)
    else:
        camera_profile = None

    horizon_model_cfg = cfg.get("horizon_model")
    horizon_model: HorizonModel
    if horizon_model_cfg is None:
        horizon_model = FlatHorizonModel()
    else:
        horizon_model = horizon_model_cfg
    horizon_profile = horizon_model.horizon_profile(lat, lon, n_az)

    horizon = sky_rgb[0]
    horizon_color = [sum(pixel[c] for pixel in horizon) / len(horizon) for c in range(3)]

    view_grid: list[list[list[float] | None]] | None = None
    ground_fraction = 1.0
    no_ground = False
    if camera_profile is not None:
        view_grid, sky_pixels_view, ground_count = sample_sky_in_camera_view(
            sky_rgb=sky_rgb,
            n_az=n_az,
            n_el=n_el,
            camera=camera_profile,
            horizon_profile=horizon_profile,
        )
        total = len(sky_pixels_view) + ground_count
        ground_fraction = (ground_count / total) if total > 0 else 0.0
        no_ground = ground_fraction <= 1e-9

    ground_pixels = _build_ground_pixels(
        dt=dt,
        lat=lat,
        lon=lon,
        atmos=atmos,
        surface=surface,
        horizon_color=horizon_color,
        ground_samples=ground_samples,
    )

    vector, meta = compute_perceptual_v1_from_buffers(
        sky_rgb,
        ground_pixels,
        view_grid=view_grid,
        no_ground=no_ground,
    )
    _, _, sun_elev_deg = solar_position(dt, lat, lon)
    sun_az = float(sky_meta.get("saz_deg", 0.0)) % 360.0
    sun_az_idx = min(int((sun_az / 360.0) * n_az), n_az - 1)
    sun_occluded = sun_elev_deg < float(horizon_profile[sun_az_idx])
    meta.update(
        {
            "sun_elev_deg": sun_elev_deg,
            "turbidity": sky_meta.get("turbidity"),
            "sza_deg": sky_meta.get("sza_deg"),
            "ground_fraction": ground_fraction,
            "horizon_elev_max_deg": max(horizon_profile) if horizon_profile else 0.0,
            "horizon_elev_mean_deg": (sum(horizon_profile) / len(horizon_profile)) if horizon_profile else 0.0,
            "sun_occluded": sun_occluded,
        }
    )
    if no_ground:
        meta["quality_flag"] = "no_ground"
    return vector, meta
