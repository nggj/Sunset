"""Provider interfaces for atmosphere/surface state ingestion."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from skycolor_locator.contracts import AtmosphereState, SurfaceState


class EarthStateProvider(Protocol):
    """Interface for retrieving atmosphere state for a place and time."""

    def get_atmosphere_state(self, dt: datetime, lat: float, lon: float) -> AtmosphereState:
        """Return atmospheric state for UTC datetime and WGS84 point."""


class SurfaceProvider(Protocol):
    """Interface for retrieving static/low-frequency surface state."""

    def get_surface_state(self, lat: float, lon: float) -> SurfaceState:
        """Return surface state for WGS84 point."""
