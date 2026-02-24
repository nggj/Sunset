"""Deterministic Earth Engine builder for tile/period surface constants.

This module is opt-in and does not import ``earthengine-api`` on import-time paths.
Callers must provide a pre-initialized ``ee`` object or configure environment credentials
used by :func:`skycolor_locator.ingest.gee_client.init_ee`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from skycolor_locator.contracts import PeriodicSurfaceConstants
from skycolor_locator.geo.tiling import tile_bounds, tile_id_for
from skycolor_locator.ingest.gee_client import GeeConfig, config_from_env, init_ee

_S2_SR = "COPERNICUS/S2_SR_HARMONIZED"
_DYNAMIC_WORLD = "GOOGLE/DYNAMICWORLD/V1"
_WORLDCOVER = "ESA/WorldCover/v200"


@dataclass(frozen=True)
class S2PeriodicConfig:
    """Configuration for deterministic periodic surface-constant construction."""

    tile_step_deg: float = 0.05
    max_cloud_pct: float = 60.0
    built_prob_threshold: float = 0.6
    reduce_scale_m: int = 10
    rgb_reflectance_white: float = 0.30
    fallback_class_rgb: dict[str, list[float]] = field(
        default_factory=lambda: {
            "ocean": [0.14, 0.38, 0.62],
            "land": [0.45, 0.40, 0.26],
            "urban": [0.50, 0.50, 0.52],
            "snow": [0.92, 0.94, 0.98],
            "desert": [0.80, 0.68, 0.42],
            "forest": [0.12, 0.35, 0.16],
        }
    )


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _lin_reflectance_to_srgb(reflectance_rgb: list[float], white: float) -> list[float]:
    """Convert reflectance RGB to display-ish sRGB using deterministic gamma mapping."""
    out: list[float] = []
    for value in reflectance_rgb:
        lin = max(0.0, min(1.0, float(value) / white))
        out.append(pow(lin, 1.0 / 2.2))
    return out


class S2PeriodicConstantsBuilder:
    """Build one tile/period periodic constants payload from Sentinel-2 + DW/WorldCover."""

    def __init__(
        self,
        ee: Any | None = None,
        gee_cfg: GeeConfig | None = None,
        cfg: S2PeriodicConfig | None = None,
    ) -> None:
        self._ee = ee
        self._gee_cfg = gee_cfg
        self._cfg = cfg or S2PeriodicConfig()

    def _ensure_ee(self) -> Any:
        if self._ee is not None:
            return self._ee
        cfg = self._gee_cfg or config_from_env()
        self._ee = init_ee(cfg)
        return self._ee

    def build(
        self,
        period_start_utc: datetime,
        period_end_utc: datetime,
        lat: float,
        lon: float,
    ) -> PeriodicSurfaceConstants:
        """Compute deterministic periodic constants for a single (tile, period)."""
        ee = self._ensure_ee()
        start = _to_utc(period_start_utc)
        end = _to_utc(period_end_utc)
        if end <= start:
            raise ValueError("period_end_utc must be greater than period_start_utc")

        lon_min, lat_min, lon_max, lat_max = tile_bounds(lat, lon, self._cfg.tile_step_deg)
        tile = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], proj=None, geodesic=False)

        s2 = (
            ee.ImageCollection(_S2_SR)
            .filterDate(ee.Date(start.isoformat()), ee.Date(end.isoformat()))
            .filterBounds(tile)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self._cfg.max_cloud_pct))
        )
        s2_scene_count = int(s2.size().getInfo())

        def _mask_clouds(img: Any) -> Any:
            scl = img.select("SCL")
            invalid = scl.eq(1)
            for code in [2, 3, 7, 8, 9, 10, 11]:
                invalid = invalid.Or(scl.eq(code))
            mask = invalid.Not()
            return img.updateMask(mask).select(["B4", "B3", "B2"]).multiply(0.0001)

        s2_composite = s2.map(_mask_clouds).median()

        dw = (
            ee.ImageCollection(_DYNAMIC_WORLD)
            .filterDate(ee.Date(start.isoformat()), ee.Date(end.isoformat()))
            .filterBounds(tile)
        )
        dw_scene_count = int(dw.size().getInfo())

        dw_empty = dw_scene_count == 0
        if not dw_empty:
            label_mode = dw.select("label").reduce(ee.Reducer.mode())
            built_prob_mean = dw.select("built").mean()
            built_mask = label_mode.eq(6).And(built_prob_mean.gte(self._cfg.built_prob_threshold))
            hist_src = label_mode
            hist_band = "label_mode"
        else:
            wc = ee.ImageCollection(_WORLDCOVER).first().select("Map")
            built_mask = wc.eq(50)
            hist_src = wc
            hist_band = "Map"

        urban_rgb_raw = (
            s2_composite.updateMask(built_mask)
            .reduceRegion(
                reducer=ee.Reducer.percentile([50]),
                geometry=tile,
                scale=self._cfg.reduce_scale_m,
                maxPixels=50_000_000,
                bestEffort=True,
            )
            .getInfo()
        )

        r = None if urban_rgb_raw is None else urban_rgb_raw.get("B4_p50")
        g = None if urban_rgb_raw is None else urban_rgb_raw.get("B3_p50")
        b = None if urban_rgb_raw is None else urban_rgb_raw.get("B2_p50")

        urban_rgb = None
        if r is not None and g is not None and b is not None:
            urban_rgb = _lin_reflectance_to_srgb([float(r), float(g), float(b)], self._cfg.rgb_reflectance_white)

        total_count = (
            hist_src.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=tile,
                scale=self._cfg.reduce_scale_m,
                maxPixels=50_000_000,
                bestEffort=True,
            )
            .get(hist_band)
            .getInfo()
        )
        built_count = (
            built_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=tile,
                scale=self._cfg.reduce_scale_m,
                maxPixels=50_000_000,
                bestEffort=True,
            )
            .get("constant")
            .getInfo()
        )

        hist = (
            hist_src.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=tile,
                scale=self._cfg.reduce_scale_m,
                maxPixels=50_000_000,
                bestEffort=True,
            )
            .get(hist_band)
            .getInfo()
        )

        landcover_mix = self._map_hist_to_surface_mix(hist=hist if isinstance(hist, dict) else {}, dw_empty=dw_empty)

        class_rgb = {k: list(v) for k, v in self._cfg.fallback_class_rgb.items()}
        if urban_rgb is not None:
            class_rgb["urban"] = urban_rgb

        valid_pixel_fraction = 0.0
        if total_count:
            valid_pixel_fraction = max(0.0, float(built_count or 0.0) / float(total_count))

        return PeriodicSurfaceConstants(
            tile_id=tile_id_for(lat, lon, self._cfg.tile_step_deg),
            period_start_utc=start,
            period_end_utc=end,
            landcover_mix=landcover_mix,
            class_rgb=class_rgb,
            meta={
                "source": "s2_dynamic_world",
                "dw_fallback_worldcover": dw_empty,
                "urban_rgb_missing": urban_rgb is None,
                "s2_scene_count": s2_scene_count,
                "dw_scene_count": dw_scene_count,
                "valid_pixel_fraction_urban": valid_pixel_fraction,
                "params": {
                    "tile_step_deg": self._cfg.tile_step_deg,
                    "max_cloud_pct": self._cfg.max_cloud_pct,
                    "built_prob_threshold": self._cfg.built_prob_threshold,
                    "reduce_scale_m": self._cfg.reduce_scale_m,
                },
            },
        )

    @staticmethod
    def _map_hist_to_surface_mix(hist: dict[str, Any], dw_empty: bool) -> dict[str, float]:
        counts: dict[int, float] = {}
        for key, value in hist.items():
            try:
                counts[int(key)] = float(value)
            except (TypeError, ValueError):
                continue

        total = sum(max(0.0, v) for v in counts.values())
        if total <= 0.0:
            return {"land": 1.0}

        def frac_dw(code: int) -> float:
            return counts.get(code, 0.0) / total

        if not dw_empty:
            ocean = frac_dw(0)
            forest = frac_dw(1)
            urban = frac_dw(6)
            snow = frac_dw(8)
            desert = frac_dw(7)
            land = frac_dw(2) + frac_dw(3) + frac_dw(4) + frac_dw(5)
        else:
            # WorldCover fallback legend: water=80 trees=10 built=50 snow=70 bare=60
            ocean = frac_dw(80)
            forest = frac_dw(10)
            urban = frac_dw(50)
            snow = frac_dw(70)
            desert = frac_dw(60)
            land = max(0.0, 1.0 - (ocean + forest + urban + snow + desert))

        mix = {
            "ocean": ocean,
            "forest": forest,
            "urban": urban,
            "snow": snow,
            "desert": desert,
            "land": land,
        }
        norm = sum(mix.values())
        if norm <= 0.0:
            return {"land": 1.0}
        return {k: v / norm for k, v in mix.items()}
