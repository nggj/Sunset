"""Optional GEE-backed SRTM horizon model (lazy Earth Engine usage)."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, pi, radians, sin
from typing import Any

from skycolor_locator.geo.tiling import tile_id_for
from skycolor_locator.ingest.gee_client import config_from_env, init_ee

_EARTH_RADIUS_M = 6_371_000.0
_SRTM = "USGS/SRTMGL1_003"


@dataclass(frozen=True)
class GeeSrtmHorizonConfig:
    """Configuration for SRTM horizon extraction."""

    max_distance_km: float = 30.0
    distance_samples_m: list[float] = field(
        default_factory=lambda: [250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 30000.0]
    )
    az_bins: int = 72
    tile_step_deg: float = 0.05


class GeeSrtmHorizonModel:
    """Horizon model based on SRTM elevation sampling in Earth Engine."""

    def __init__(self, cfg: GeeSrtmHorizonConfig | None = None) -> None:
        self._cfg = cfg or GeeSrtmHorizonConfig()
        self._ee: Any | None = None
        self._cache: dict[str, list[float]] = {}

    def _ensure_ee(self) -> Any:
        if self._ee is None:
            self._ee = init_ee(config_from_env())
        return self._ee

    def horizon_profile(self, lat: float, lon: float, az_bins: int) -> list[float]:
        """Return azimuth-binned horizon elevation profile in degrees."""
        if az_bins <= 0:
            raise ValueError("az_bins must be positive")
        tile = tile_id_for(lat, lon, self._cfg.tile_step_deg)
        cache_key = f"{tile}:{az_bins}"
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        ee = self._ensure_ee()
        dem = ee.Image(_SRTM).select("elevation")
        center = ee.Geometry.Point([lon, lat])

        dist_samples = [d for d in self._cfg.distance_samples_m if d <= self._cfg.max_distance_km * 1000.0]
        if not dist_samples:
            dist_samples = [self._cfg.max_distance_km * 1000.0]

        features: list[Any] = []
        for az_idx in range(az_bins):
            az_deg = (az_idx / az_bins) * 360.0
            az_rad = radians(az_deg)
            for dist_m in dist_samples:
                dlat = (dist_m * cos(az_rad)) / 111_320.0
                dlon = (dist_m * sin(az_rad)) / (111_320.0 * max(0.1, cos(radians(lat))))
                pt = ee.Geometry.Point([lon + dlon, lat + dlat])
                features.append(ee.Feature(pt, {"az_idx": az_idx, "distance_m": dist_m}))

        fc = ee.FeatureCollection(features)
        sampled = dem.sampleRegions(collection=fc, scale=30, geometries=False)
        rows = sampled.getInfo().get("features", [])

        view_elev = dem.reduceRegion(
            reducer=ee.Reducer.first(), geometry=center, scale=30, bestEffort=True
        ).get("elevation").getInfo()
        elev_view = float(view_elev or 0.0)

        by_az: dict[int, float] = {i: -90.0 for i in range(az_bins)}
        for row in rows:
            props = row.get("properties", {})
            az_idx = int(props.get("az_idx", 0))
            dist_m = float(props.get("distance_m", 1.0))
            elev_sample = float(props.get("elevation", elev_view))
            curvature_drop_m = (dist_m * dist_m) / (2.0 * _EARTH_RADIUS_M)
            angle_deg = atan2((elev_sample - elev_view - curvature_drop_m), dist_m) * 180.0 / pi
            by_az[az_idx] = max(by_az[az_idx], angle_deg)

        profile = [float(by_az[i]) for i in range(az_bins)]
        self._cache[cache_key] = list(profile)
        return profile

    def meta(self) -> dict[str, Any]:
        """Return metadata for SRTM horizon model settings."""
        return {
            "model": "srtm",
            "max_distance_km": self._cfg.max_distance_km,
            "distance_samples_m": list(self._cfg.distance_samples_m),
            "az_bins": self._cfg.az_bins,
        }
