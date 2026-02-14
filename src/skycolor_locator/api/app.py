"""FastAPI app exposing signature generation and search endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from skycolor_locator.index.bruteforce import BruteforceIndex
from skycolor_locator.ingest.mock_providers import MockEarthStateProvider, MockSurfaceProvider
from skycolor_locator.orchestrate.batch import GridSpec as BatchGridSpec
from skycolor_locator.orchestrate.batch import generate_lat_lon_grid
from skycolor_locator.signature.core import compute_color_signature

_MODEL_VERSION = "mvp-v1"


class SignatureRequest(BaseModel):
    """Request schema for generating one signature."""

    time_utc: datetime
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)


class ColorSignatureResponse(BaseModel):
    """Response schema aligned with ColorSignature contract."""

    hue_bins: list[float]
    sky_hue_hist: list[float]
    ground_hue_hist: list[float]
    signature: list[float]
    meta: dict[str, Any]
    uncertainty_score: float
    quality_flags: list[str]


class GridSpecRequest(BaseModel):
    """Grid bounds used to build/reuse a searchable index."""

    lat_min: float = Field(default=-10.0, ge=-90.0, le=90.0)
    lat_max: float = Field(default=10.0, ge=-90.0, le=90.0)
    lon_min: float = Field(default=100.0, ge=-180.0, le=180.0)
    lon_max: float = Field(default=120.0, ge=-180.0, le=180.0)
    step_deg: float = Field(default=5.0, gt=0.0)
    max_points: int = Field(default=2500, ge=1, le=20000)


class TargetPointRequest(BaseModel):
    """Optional target location/time used to generate target signature."""

    time_utc: datetime | None = None
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)


class SearchRequest(BaseModel):
    """Request schema for color-signature search."""

    target_signature: list[float] | None = None
    target: TargetPointRequest | None = None
    time_utc: datetime | None = None
    time_bucket_minutes: int = Field(default=10, ge=1, le=1440)
    top_k: int = Field(default=3, ge=1, le=50)
    filters: dict[str, Any] | None = None
    grid_spec: GridSpecRequest = Field(default_factory=GridSpecRequest)


class SearchCandidate(BaseModel):
    """One search result candidate."""

    key: str
    distance: float


class SearchResponse(BaseModel):
    """Search response payload."""

    candidates: list[SearchCandidate]


class CachedIndexEntry:
    """Cached index payload with metadata."""

    def __init__(self, index: BruteforceIndex, built_at: datetime, metadata: dict[str, Any]) -> None:
        self.index = index
        self.built_at = built_at
        self.metadata = metadata


class IndexStore:
    """In-memory TTL cache for prebuilt search indices."""

    def __init__(self, ttl_seconds: int = 600, max_entries: int = 8) -> None:
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_entries = max_entries
        self._entries: dict[str, CachedIndexEntry] = {}
        self._lock = Lock()
        self.build_count = 0

    def _evict_expired(self, now: datetime) -> None:
        expired = [k for k, v in self._entries.items() if now - v.built_at > self.ttl]
        for key in expired:
            del self._entries[key]

    def _evict_lru(self) -> None:
        if len(self._entries) <= self.max_entries:
            return
        oldest = sorted(self._entries.items(), key=lambda kv: kv[1].built_at)[0][0]
        del self._entries[oldest]


    def reset(self) -> None:
        """Reset cache entries and build counter (test helper)."""
        with self._lock:
            self._entries.clear()
            self.build_count = 0

    def get_or_build(self, key: str, builder: Any) -> tuple[BruteforceIndex, bool]:
        """Return cached index or build and cache one for key."""
        now = datetime.now(timezone.utc)
        with self._lock:
            self._evict_expired(now)
            if key in self._entries:
                return self._entries[key].index, True

            index, metadata = builder()
            self.build_count += 1
            self._entries[key] = CachedIndexEntry(index=index, built_at=now, metadata=metadata)
            self._evict_lru()
            return index, False


def _normalize_time(dt: datetime | None) -> datetime:
    """Normalize optional datetime to timezone-aware UTC value."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _bucket_time(dt: datetime, bucket_minutes: int) -> datetime:
    """Round down datetime to deterministic bucket boundary."""
    minute_bucket = (dt.minute // bucket_minutes) * bucket_minutes
    return dt.replace(minute=minute_bucket, second=0, microsecond=0)


def _grid_key(spec: GridSpecRequest) -> str:
    """Return stable string key fragment for grid specification."""
    return (
        f"{spec.lat_min:.6f}:{spec.lat_max:.6f}:{spec.lon_min:.6f}:"
        f"{spec.lon_max:.6f}:{spec.step_deg:.6f}:{spec.max_points}"
    )


def _signature_bins(payload: SearchRequest) -> int:
    """Infer bin count from target signature if available, else default."""
    if payload.target_signature is not None and len(payload.target_signature) % 2 == 0:
        return len(payload.target_signature) // 2
    return 36


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(title="Skycolor Locator API", version="0.1.0")

    earth_provider = MockEarthStateProvider()
    surface_provider = MockSurfaceProvider()
    index_store = IndexStore(ttl_seconds=600, max_entries=8)
    app.state.index_store = index_store

    @app.post("/signature", response_model=ColorSignatureResponse)
    def post_signature(payload: SignatureRequest) -> ColorSignatureResponse:
        """Generate one color signature for input time/location."""
        dt = _normalize_time(payload.time_utc)
        atmos = earth_provider.get_atmosphere_state(dt, payload.lat, payload.lon)
        surface = surface_provider.get_surface_state(payload.lat, payload.lon)
        signature = compute_color_signature(dt, payload.lat, payload.lon, atmos, surface)
        return ColorSignatureResponse(**signature.to_dict())

    @app.post("/search", response_model=SearchResponse)
    def post_search(payload: SearchRequest) -> SearchResponse:
        """Search candidate locations by target signature similarity."""
        base_dt = _normalize_time(payload.time_utc)
        bucketed_dt = _bucket_time(base_dt, payload.time_bucket_minutes)
        bins = _signature_bins(payload)

        grid_spec = BatchGridSpec(
            lat_min=payload.grid_spec.lat_min,
            lat_max=payload.grid_spec.lat_max,
            lon_min=payload.grid_spec.lon_min,
            lon_max=payload.grid_spec.lon_max,
            step_deg=payload.grid_spec.step_deg,
        )
        points = generate_lat_lon_grid(grid_spec)
        if len(points) > payload.grid_spec.max_points:
            raise HTTPException(status_code=400, detail="grid_spec produced too many points")

        cache_key = f"{bucketed_dt.isoformat()}|{bins}|{_grid_key(payload.grid_spec)}|{_MODEL_VERSION}"

        def builder() -> tuple[BruteforceIndex, dict[str, Any]]:
            index = BruteforceIndex(mode="cosine")
            keys: list[str] = []
            vectors: list[list[float]] = []
            metadatas: list[dict[str, Any]] = []
            cfg = {"bins": bins, "n_az": 24, "n_el": 12}

            for lat, lon in points:
                atmos = earth_provider.get_atmosphere_state(bucketed_dt, lat, lon)
                surface = surface_provider.get_surface_state(lat, lon)
                sig = compute_color_signature(bucketed_dt, lat, lon, atmos, surface, config=cfg)

                keys.append(f"lat={lat:.3f},lon={lon:.3f}")
                vectors.append(sig.signature)
                metadatas.append(
                    {
                        "time_utc": bucketed_dt.isoformat(),
                        "landcover_tags": list(surface.landcover_mix.keys()) or [surface.surface_class.value],
                        "lat": lat,
                        "lon": lon,
                    }
                )

            index.add(keys, vectors, metadatas=metadatas)
            return index, {"points": len(points), "bucket_time_utc": bucketed_dt.isoformat()}

        index, _cache_hit = index_store.get_or_build(cache_key, builder)

        if payload.target_signature is not None:
            target_signature = payload.target_signature
        elif payload.target is not None:
            target_dt = _normalize_time(payload.target.time_utc or bucketed_dt)
            target_atmos = earth_provider.get_atmosphere_state(target_dt, payload.target.lat, payload.target.lon)
            target_surface = surface_provider.get_surface_state(payload.target.lat, payload.target.lon)
            target_sig = compute_color_signature(
                target_dt,
                payload.target.lat,
                payload.target.lon,
                target_atmos,
                target_surface,
                config={"bins": bins, "n_az": 24, "n_el": 12},
            )
            target_signature = target_sig.signature
        else:
            raise HTTPException(status_code=400, detail="Provide target_signature or target")

        result = index.query(target_signature, payload.top_k, filters=payload.filters)
        return SearchResponse(candidates=[SearchCandidate(key=k, distance=d) for k, d in result])

    return app


app = create_app()
