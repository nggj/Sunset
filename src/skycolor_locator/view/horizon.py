"""Horizon model interfaces and baseline implementations."""

from __future__ import annotations

from typing import Any, Protocol


class HorizonModel(Protocol):
    """Interface for azimuth-binned horizon elevation profiles."""

    def horizon_profile(self, lat: float, lon: float, az_bins: int) -> list[float]:
        """Return horizon elevation degrees for each azimuth bin."""

    def meta(self) -> dict[str, Any]:
        """Return metadata describing horizon model implementation."""


class FlatHorizonModel:
    """Flat-horizon model returning 0° elevation for every azimuth."""

    def horizon_profile(self, lat: float, lon: float, az_bins: int) -> list[float]:
        """Return a zero-elevation profile of length az_bins."""
        if az_bins <= 0:
            raise ValueError("az_bins must be positive")
        return [0.0] * az_bins

    def meta(self) -> dict[str, Any]:
        """Return model metadata."""
        return {"model": "flat"}
