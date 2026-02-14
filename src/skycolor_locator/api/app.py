"""FastAPI app exposing signature generation and search endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from skycolor_locator.index.bruteforce import BruteforceIndex
from skycolor_locator.ingest.mock_providers import MockEarthStateProvider, MockSurfaceProvider
from skycolor_locator.signature.core import compute_color_signature


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


class SearchRequest(BaseModel):
    """Request schema for color-signature search."""

    target_signature: list[float]
    time_utc: datetime | None = None
    top_k: int = Field(default=3, ge=1, le=50)


class SearchCandidate(BaseModel):
    """One search result candidate."""

    key: str
    distance: float


class SearchResponse(BaseModel):
    """Search response payload."""

    candidates: list[SearchCandidate]


def _normalize_time(dt: datetime | None) -> datetime:
    """Normalize optional datetime to timezone-aware UTC value."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(title="Skycolor Locator API", version="0.1.0")

    earth_provider = MockEarthStateProvider()
    surface_provider = MockSurfaceProvider()

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
        dt = _normalize_time(payload.time_utc)

        # Small deterministic candidate grid for MVP.
        candidates = [
            ("seoul", 37.5665, 126.9780),
            ("sydney", -33.8688, 151.2093),
            ("nairobi", -1.2921, 36.8219),
            ("london", 51.5074, -0.1278),
            ("lima", -12.0464, -77.0428),
        ]

        index = BruteforceIndex(mode="cosine")
        keys: list[str] = []
        vectors: list[list[float]] = []
        for key, lat, lon in candidates:
            atmos = earth_provider.get_atmosphere_state(dt, lat, lon)
            surface = surface_provider.get_surface_state(lat, lon)
            sig = compute_color_signature(dt, lat, lon, atmos, surface)
            keys.append(key)
            vectors.append(sig.signature)

        index.add(keys, vectors)
        result = index.query(payload.target_signature, payload.top_k)
        return SearchResponse(candidates=[SearchCandidate(key=k, distance=d) for k, d in result])

    return app


app = create_app()
