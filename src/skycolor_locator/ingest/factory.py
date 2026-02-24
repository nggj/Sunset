"""Provider factory functions with lazy GEE imports."""

from __future__ import annotations

import os

from skycolor_locator.ingest.interfaces import EarthStateProvider, SurfaceProvider
from skycolor_locator.ingest.mock_providers import MockEarthStateProvider, MockSurfaceProvider


def _resolve_mode(mode: str | None) -> str:
    """Resolve provider mode from argument or environment."""
    raw = mode or os.getenv("SKYCOLOR_PROVIDER_MODE", "mock")
    resolved = raw.strip().lower()
    if resolved not in {"mock", "gee"}:
        raise ValueError("provider mode must be one of: mock, gee")
    return resolved


def create_earth_provider(mode: str | None = None) -> EarthStateProvider:
    """Create an EarthState provider for the selected mode."""
    resolved = _resolve_mode(mode)
    if resolved == "mock":
        return MockEarthStateProvider()

    from skycolor_locator.ingest.gee_providers import GeeEarthStateProvider

    return GeeEarthStateProvider()


def create_surface_provider(mode: str | None = None) -> SurfaceProvider:
    """Create a Surface provider for the selected mode."""
    resolved = _resolve_mode(mode)
    if resolved == "mock":
        return MockSurfaceProvider()

    from skycolor_locator.ingest.gee_providers import GeeSurfaceProvider

    return GeeSurfaceProvider()
