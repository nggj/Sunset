"""FastAPI app exposing signature generation and search endpoints."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

from skycolor_locator.index.bruteforce import BruteforceIndex
from skycolor_locator.index.store import IndexCacheKey, IndexStore
from skycolor_locator.ingest.mock_providers import MockEarthStateProvider, MockSurfaceProvider
from skycolor_locator.orchestrate.batch import GridSpec, generate_lat_lon_grid
from skycolor_locator.signature.core import compute_color_signature

_MODEL_VERSION = "mvp-v1"
_DEFAULT_GRID_SPEC = GridSpec(
    lat_min=-60.0,
    lat_max=60.0,
    lon_min=-180.0,
    lon_max=180.0,
    step_deg=60.0,
    max_points=500,
)


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
    """Request-level grid specification for candidate generation."""

    lat_min: float = Field(ge=-90.0, le=90.0)
    lat_max: float = Field(ge=-90.0, le=90.0)
    lon_min: float = Field(ge=-180.0, le=180.0)
    lon_max: float = Field(ge=-180.0, le=180.0)
    step_deg: float = Field(gt=0.0, le=30.0)
    max_points: int | None = Field(default=5_000, ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "GridSpecRequest":
        """Validate coordinate bounds and ordering."""
        if self.lat_min > self.lat_max:
            raise ValueError("lat_min must be <= lat_max")
        if self.lon_min > self.lon_max:
            raise ValueError("lon_min must be <= lon_max")
        return self

    def to_spec(self) -> GridSpec:
        """Convert API schema into orchestration grid spec."""
        return GridSpec(
            lat_min=self.lat_min,
            lat_max=self.lat_max,
            lon_min=self.lon_min,
            lon_max=self.lon_max,
            step_deg=self.step_deg,
            max_points=self.max_points,
        )


class SearchFilters(BaseModel):
    """Optional post-query filters for candidate metadata."""

    surface_classes: list[str] | None = None


class SearchRequest(BaseModel):
    """Request schema for color-signature search."""

    target_signature: list[float] | None = None
    target_time_utc: datetime | None = None
    target_lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    target_lon: float | None = Field(default=None, ge=-180.0, le=180.0)

    grid_spec: GridSpecRequest = Field(
        default_factory=lambda: GridSpecRequest(**_DEFAULT_GRID_SPEC.__dict__)
    )
    time_utc: datetime | None = None
    window_start_utc: datetime | None = None
    window_end_utc: datetime | None = None
    bucket_minutes: int = Field(default=60, ge=1, le=24 * 60)

    top_k: int = Field(default=3, ge=1, le=50)
    filters: SearchFilters | None = None

    @model_validator(mode="after")
    def validate_target_source(self) -> "SearchRequest":
        """Require either target signature or target lat/lon/time inputs."""
        has_signature = self.target_signature is not None
        has_target_point = (
            self.target_time_utc is not None
            and self.target_lat is not None
            and self.target_lon is not None
        )
        if not has_signature and not has_target_point:
            raise ValueError(
                "Provide target_signature, or target_time_utc+target_lat+target_lon to generate one."
            )
        if (
            self.window_start_utc
            and self.window_end_utc
            and self.window_start_utc > self.window_end_utc
        ):
            raise ValueError("window_start_utc must be <= window_end_utc")
        return self


class SearchCandidate(BaseModel):
    """One search result candidate."""

    key: str
    distance: float


class SearchResponse(BaseModel):
    """Search response payload."""

    candidates: list[SearchCandidate]
    cache_hit: bool
    build_count: int


def _normalize_time(dt: datetime | None) -> datetime:
    """Normalize optional datetime to timezone-aware UTC value."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_grid_spec(payload_spec: GridSpecRequest) -> GridSpec:
    """Convert and validate request grid spec."""
    return payload_spec.to_spec()


def _grid_hash(spec: GridSpec) -> str:
    """Compute deterministic hash for a grid specification."""
    encoded = json.dumps(
        {
            "lat_min": spec.lat_min,
            "lat_max": spec.lat_max,
            "lon_min": spec.lon_min,
            "lon_max": spec.lon_max,
            "step_deg": spec.step_deg,
            "max_points": spec.max_points,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolve_time_bucket(payload: SearchRequest) -> tuple[datetime, str]:
    """Resolve query time and bucket string from point or window inputs."""
    if payload.window_start_utc is not None:
        start = _normalize_time(payload.window_start_utc)
        end = _normalize_time(payload.window_end_utc or payload.window_start_utc)
        midpoint = start + (end - start) / 2
        bucket_start = start - timedelta(
            minutes=start.minute % payload.bucket_minutes,
            seconds=start.second,
            microseconds=start.microsecond,
        )
        return midpoint, (
            f"window:{bucket_start.isoformat()}:{int((end - start).total_seconds())}:"
            f"{payload.bucket_minutes}m"
        )

    dt = _normalize_time(payload.time_utc)
    bucket_start = dt - timedelta(
        minutes=dt.minute % payload.bucket_minutes,
        seconds=dt.second,
        microseconds=dt.microsecond,
    )
    return dt, f"time:{bucket_start.isoformat()}:{payload.bucket_minutes}m"


def _candidate_passes_filters(metadata: dict[str, Any], filters: SearchFilters | None) -> bool:
    """Apply optional metadata-based filters to a candidate."""
    if filters is None:
        return True
    if filters.surface_classes is not None:
        surface_class = str(metadata.get("surface_class", ""))
        if surface_class not in filters.surface_classes:
            return False
    return True


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(title="Skycolor Locator API", version="0.1.0")

    earth_provider = MockEarthStateProvider()
    surface_provider = MockSurfaceProvider()
    index_store = IndexStore(ttl_seconds=600, max_entries=16)

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
        query_time, time_bucket = _resolve_time_bucket(payload)
        grid_spec = _normalize_grid_spec(payload.grid_spec)

        if payload.target_signature is not None:
            target_signature = payload.target_signature
        else:
            target_dt = _normalize_time(payload.target_time_utc)
            assert payload.target_lat is not None and payload.target_lon is not None
            target_atmos = earth_provider.get_atmosphere_state(
                target_dt, payload.target_lat, payload.target_lon
            )
            target_surface = surface_provider.get_surface_state(
                payload.target_lat, payload.target_lon
            )
            target_signature = compute_color_signature(
                target_dt,
                payload.target_lat,
                payload.target_lon,
                target_atmos,
                target_surface,
            ).signature

        if len(target_signature) % 2 != 0:
            raise ValueError("target_signature length must be even")
        bins = len(target_signature) // 2

        cache_key = IndexCacheKey(
            time_bucket=time_bucket,
            bins=bins,
            grid_spec_hash=_grid_hash(grid_spec),
            model_version=_MODEL_VERSION,
        )

        def builder() -> tuple[BruteforceIndex, dict[str, dict[str, Any]]]:
            index = BruteforceIndex(mode="cosine")
            metadata_by_key: dict[str, dict[str, Any]] = {}
            keys: list[str] = []
            vectors: list[list[float]] = []
            for lat, lon in generate_lat_lon_grid(grid_spec):
                atmos = earth_provider.get_atmosphere_state(query_time, lat, lon)
                surface = surface_provider.get_surface_state(lat, lon)
                signature = compute_color_signature(query_time, lat, lon, atmos, surface)
                key = f"lat={lat:.3f},lon={lon:.3f},t={query_time.isoformat()}"
                keys.append(key)
                vectors.append(signature.signature)
                metadata_by_key[key] = {
                    "lat": lat,
                    "lon": lon,
                    "time_utc": query_time.isoformat(),
                    "surface_class": surface.surface_class.value,
                    "landcover_tags": sorted(surface.landcover_mix.keys()),
                }
            index.add(keys, vectors)
            return index, metadata_by_key

        entry, was_built = index_store.get_or_build(cache_key, builder)
        all_ranked = entry.index.query(
            target_signature, top_k=max(payload.top_k * 10, payload.top_k)
        )
        filtered = [
            (key, dist)
            for key, dist in all_ranked
            if _candidate_passes_filters(entry.metadata_by_key[key], payload.filters)
        ]
        filtered.sort(key=lambda item: (item[1], item[0]))
        selected = filtered[: payload.top_k]

        return SearchResponse(
            candidates=[SearchCandidate(key=k, distance=d) for k, d in selected],
            cache_hit=not was_built,
            build_count=index_store.build_count,
        )

    return app


app = create_app()
