"""
Real-data providers backed by Google Earth Engine (GEE).

Implements:
- EarthStateProvider.get_atmosphere_state(dt, lat, lon) -> AtmosphereState
- SurfaceProvider.get_surface_state(lat, lon) -> SurfaceState

Important:
- Must be opt-in. Default app/tests should continue using mock_providers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from math import exp
from typing import Any

from skycolor_locator.contracts import AtmosphereState, SurfaceClass, SurfaceState
from skycolor_locator.ingest.cache import LRUCache
from skycolor_locator.ingest.interfaces import EarthStateProvider, SurfaceProvider
from skycolor_locator.ingest.gee_client import GeeConfig, config_from_env, init_ee

# ERA5 ozone conversion: DU = total_column_ozone / 2.1415E-5 (kg/m^2 per DU)
_DU_KG_M2 = 2.1415e-5

# Dataset IDs (Earth Engine)
_ERA5 = "ECMWF/ERA5/HOURLY"
_CAMS = "ECMWF/CAMS/NRT"
_WORLDCOVER = "ESA/WorldCover/v200"
_MCD43A3 = "MODIS/061/MCD43A3"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("dt must be timezone-aware")
    return dt.astimezone(timezone.utc)


def _nearest_image(ee: Any, collection_id: str, target_dt: datetime, window: timedelta, extra_filter: Any | None = None) -> Any | None:
    """
    Return ee.Image nearest to target_dt inside [target_dt-window, target_dt+window].

    Fully server-side selection by adding a 'time_diff' property.
    """
    start = ee.Date((target_dt - window).isoformat())
    end = ee.Date((target_dt + window).isoformat())
    ic = ee.ImageCollection(collection_id).filterDate(start, end)
    if extra_filter is not None:
        ic = ic.filter(extra_filter)

    target_ms = ee.Date(target_dt.isoformat()).millis()

    def add_diff(img: Any) -> Any:
        diff = img.date().millis().subtract(target_ms).abs()
        return img.set("time_diff", diff)

    # If collection empty, first() will be a "null" object; we handle by a try-get later.
    best = ic.map(add_diff).sort("time_diff").first()
    return best


def _reduce_point_first(ee: Any, img: Any, lat: float, lon: float, bands: list[str], scale_m: int) -> dict[str, float | None]:
    """
    Reduce a single pixel at a point using Reducer.first().
    Returns a dict {band: value or None}.
    """
    pt = ee.Geometry.Point([float(lon), float(lat)])
    # Note: reduceRegion returns raw values; scales must be applied manually if needed.
    d = (
        img.select(bands)
        .reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=pt,
            scale=scale_m,
            maxPixels=1_000_000,
            bestEffort=True,
        )
        .getInfo()
    )
    out: dict[str, float | None] = {}
    for b in bands:
        v = None if d is None else d.get(b)
        out[b] = None if v is None else float(v)
    return out


def _relative_humidity_from_t_td(temp_k: float, dewpoint_k: float) -> float:
    """
    Compute RH (0..1) from temperature and dewpoint in Kelvin.
    Uses Magnus approximation (good enough for features).
    """
    t = temp_k - 273.15
    td = dewpoint_k - 273.15
    a = 17.625
    b = 243.04
    # Use exp(a*T/(b+T)) proportional to saturation vapor pressure.
    sat = exp(a * t / (b + t))
    act = exp(a * td / (b + td))
    return _clamp(act / max(sat, 1e-12), 0.0, 1.0)


@dataclass(frozen=True)
class GeeAtmosphereConfig:
    era5_window_hours: int = 2
    cams_window_hours: int = 24
    # Scales chosen close to dataset native resolution to avoid expensive operations.
    era5_scale_m: int = 30_000
    cams_scale_m: int = 50_000

    # COD heuristic parameters (optional)
    cod_liquid_factor: float = 125.0
    cod_ice_factor: float = 60.0
    cod_max: float = 120.0


class GeeEarthStateProvider(EarthStateProvider):
    """Earth Engine-backed atmosphere provider (ERA5 + CAMS NRT)."""

    def __init__(
        self,
        gee: Any | None = None,
        cfg: GeeConfig | None = None,
        cache_size: int = 256,
        atmos_cfg: GeeAtmosphereConfig | None = None,
    ) -> None:
        self._cfg = cfg
        self._ee = gee
        self._cache: LRUCache[tuple[str, int, int], AtmosphereState] = LRUCache(cache_size)
        self._atmos_cfg = atmos_cfg or GeeAtmosphereConfig()

    def _ensure_ee(self) -> Any:
        if self._ee is not None:
            return self._ee
        cfg = self._cfg or config_from_env()
        self._ee = init_ee(cfg)
        return self._ee

    def get_atmosphere_state(self, dt: datetime, lat: float, lon: float) -> AtmosphereState:
        dt_utc = _to_utc(dt)
        key = (dt_utc.isoformat(), int(lat * 100), int(lon * 100))

        def factory() -> AtmosphereState:
            ee = self._ensure_ee()

            missing = False

            # --- ERA5 ---
            era5_img = _nearest_image(
                ee,
                _ERA5,
                dt_utc,
                window=timedelta(hours=self._atmos_cfg.era5_window_hours),
            )

            cloud_fraction = None
            ozone_du = None
            pressure_hpa = None
            humidity = None
            cloud_optical_depth = None
            cloud_ice_fraction = None

            if era5_img is not None:
                era5_bands = [
                    "total_cloud_cover",
                    "total_column_ozone",
                    "surface_pressure",
                    "temperature_2m",
                    "dewpoint_temperature_2m",
                    "total_column_cloud_liquid_water",
                    "total_column_cloud_ice_water",
                ]
                vals = _reduce_point_first(ee, era5_img, lat, lon, era5_bands, scale_m=self._atmos_cfg.era5_scale_m)

                cf = vals.get("total_cloud_cover")
                if cf is not None:
                    cloud_fraction = _clamp(cf, 0.0, 1.0)

                tco = vals.get("total_column_ozone")
                if tco is not None:
                    ozone_du = float(tco) / _DU_KG_M2

                sp = vals.get("surface_pressure")
                if sp is not None:
                    pressure_hpa = float(sp) / 100.0

                t2m = vals.get("temperature_2m")
                td2m = vals.get("dewpoint_temperature_2m")
                if t2m is not None and td2m is not None:
                    humidity = _relative_humidity_from_t_td(t2m, td2m)

                # Heuristic COD from cloud water paths (kg/m^2)
                lwp = vals.get("total_column_cloud_liquid_water")
                iwp = vals.get("total_column_cloud_ice_water")
                if lwp is not None or iwp is not None:
                    lwp_v = 0.0 if lwp is None else max(0.0, float(lwp))
                    iwp_v = 0.0 if iwp is None else max(0.0, float(iwp))
                    total_wp = lwp_v + iwp_v
                    if total_wp > 0.0:
                        cloud_ice_fraction = _clamp(iwp_v / total_wp, 0.0, 1.0)
                    cod = self._atmos_cfg.cod_liquid_factor * lwp_v + self._atmos_cfg.cod_ice_factor * iwp_v
                    cloud_optical_depth = _clamp(cod, 0.0, self._atmos_cfg.cod_max)

            # --- CAMS NRT (AOD 550) ---
            cams_img = _nearest_image(
                ee,
                _CAMS,
                dt_utc,
                window=timedelta(hours=self._atmos_cfg.cams_window_hours),
                # Optional deterministic choice: pick model_initialization_hour == 0 only
                # extra_filter=ee.Filter.eq("model_initialization_hour", 0),
            )

            aod = None
            if cams_img is not None:
                cams_vals = _reduce_point_first(
                    ee,
                    cams_img,
                    lat,
                    lon,
                    ["total_aerosol_optical_depth_at_550nm_surface"],
                    scale_m=self._atmos_cfg.cams_scale_m,
                )
                aod = cams_vals.get("total_aerosol_optical_depth_at_550nm_surface")
                if aod is not None:
                    aod = _clamp(float(aod), 0.0, 5.0)

            # --- Fill required fields with fallback if missing ---
            if cloud_fraction is None:
                cloud_fraction = 0.5
                missing = True
            if aod is None:
                aod = 0.1
                missing = True
            if ozone_du is None:
                ozone_du = 300.0
                missing = True

            return AtmosphereState(
                cloud_fraction=float(cloud_fraction),
                aerosol_optical_depth=float(aod),
                total_ozone_du=float(ozone_du),
                humidity=humidity,
                pressure_hpa=pressure_hpa,
                cloud_optical_depth=cloud_optical_depth,
                cloud_ice_fraction=cloud_ice_fraction,
                missing_realtime=missing,
            )

        return self._cache.get(key, factory)


@dataclass(frozen=True)
class GeeSurfaceConfig:
    # Tile radius used for class mix (meters). Spec mentions ~5km tiles; radius=2500m matches.
    radius_m: int = 2500
    # Sampling scale for WorldCover histogram. 10m is accurate but heavier; 50~100m is fast.
    worldcover_scale_m: int = 100
    # Albedo reference date: keep deterministic since interface lacks time.
    albedo_date: date = date(2025, 6, 15)
    albedo_scale_m: int = 500
    albedo_band: str = "Albedo_WSA_shortwave"
    albedo_raw_scale: float = 0.001


class GeeSurfaceProvider(SurfaceProvider):
    """Earth Engine-backed surface provider (WorldCover + MODIS albedo)."""

    def __init__(
        self,
        gee: Any | None = None,
        cfg: GeeConfig | None = None,
        cache_size: int = 512,
        surf_cfg: GeeSurfaceConfig | None = None,
    ) -> None:
        self._cfg = cfg
        self._ee = gee
        self._cache: LRUCache[tuple[int, int], SurfaceState] = LRUCache(cache_size)
        self._surf_cfg = surf_cfg or GeeSurfaceConfig()

    def _ensure_ee(self) -> Any:
        if self._ee is not None:
            return self._ee
        cfg = self._cfg or config_from_env()
        self._ee = init_ee(cfg)
        return self._ee

    def get_surface_state(self, lat: float, lon: float) -> SurfaceState:
        key = (int(lat * 100), int(lon * 100))

        def factory() -> SurfaceState:
            ee = self._ensure_ee()

            # --- Landcover mix from WorldCover ---
            pt = ee.Geometry.Point([float(lon), float(lat)])
            region = pt.buffer(self._surf_cfg.radius_m)

            wc_img = ee.ImageCollection(_WORLDCOVER).first().select("Map")
            # frequencyHistogram gives {class_value: count}
            hist = (
                wc_img.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=region,
                    scale=self._surf_cfg.worldcover_scale_m,
                    maxPixels=10_000_000,
                    bestEffort=True,
                )
                .get("Map")
                .getInfo()
            )

            # Map WorldCover values -> our SurfaceClass buckets
            # WorldCover legend: 10 trees, 50 built-up, 60 bare, 70 snow/ice, 80 water, etc.
            counts: dict[int, float] = {}
            if isinstance(hist, dict):
                for k, v in hist.items():
                    try:
                        counts[int(k)] = float(v)
                    except Exception:
                        continue

            total = sum(counts.values())
            if total <= 0:
                # Fallback
                return SurfaceState(surface_class=SurfaceClass.LAND, dominant_albedo=0.25, landcover_mix={"land": 1.0})

            def frac(code: int) -> float:
                return counts.get(code, 0.0) / total

            ocean_f = frac(80)
            snow_f = frac(70)
            urban_f = frac(50)
            desert_f = frac(60)
            forest_f = frac(10)

            # Everything else -> land
            known = ocean_f + snow_f + urban_f + desert_f + forest_f
            land_f = _clamp(1.0 - known, 0.0, 1.0)

            landcover_mix = {
                SurfaceClass.OCEAN.value: ocean_f,
                SurfaceClass.SNOW.value: snow_f,
                SurfaceClass.URBAN.value: urban_f,
                SurfaceClass.DESERT.value: desert_f,
                SurfaceClass.FOREST.value: forest_f,
                SurfaceClass.LAND.value: land_f,
            }

            # Pick dominant class (with a couple of priority rules)
            surface_class = max(
                [
                    (SurfaceClass.OCEAN, ocean_f),
                    (SurfaceClass.SNOW, snow_f),
                    (SurfaceClass.URBAN, urban_f),
                    (SurfaceClass.DESERT, desert_f),
                    (SurfaceClass.FOREST, forest_f),
                    (SurfaceClass.LAND, land_f),
                ],
                key=lambda x: x[1],
            )[0]

            # --- Albedo from MODIS MCD43A3 ---
            # Deterministic by fixed albedo_date (since interface has no dt)
            d0 = datetime(self._surf_cfg.albedo_date.year, self._surf_cfg.albedo_date.month, self._surf_cfg.albedo_date.day, tzinfo=timezone.utc)
            window = timedelta(days=8)
            albedo_img = _nearest_image(ee, _MCD43A3, d0, window=window)
            albedo = None
            if albedo_img is not None:
                vals = _reduce_point_first(
                    ee,
                    albedo_img,
                    lat,
                    lon,
                    [self._surf_cfg.albedo_band],
                    scale_m=self._surf_cfg.albedo_scale_m,
                )
                raw = vals.get(self._surf_cfg.albedo_band)
                if raw is not None:
                    scaled = float(raw) * self._surf_cfg.albedo_raw_scale
                    # Basic sanity clamp (MODIS fill values can exist)
                    if 0.0 <= scaled <= 1.5:
                        albedo = _clamp(scaled, 0.0, 1.0)

            if albedo is None:
                # Fallback by class
                albedo = {
                    SurfaceClass.OCEAN: 0.08,
                    SurfaceClass.SNOW: 0.7,
                    SurfaceClass.DESERT: 0.45,
                    SurfaceClass.URBAN: 0.2,
                    SurfaceClass.FOREST: 0.15,
                    SurfaceClass.LAND: 0.25,
                }.get(surface_class, 0.25)

            return SurfaceState(surface_class=surface_class, dominant_albedo=float(albedo), landcover_mix=landcover_mix)

        return self._cache.get(key, factory)
