"""Core data contracts for the Skycolor Locator MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isclose
from typing import Any


def _to_float_list(values: object) -> list[float]:
    """Convert list-like or numpy-like values to a list of floats."""
    if hasattr(values, "tolist"):
        raw_values = values.tolist()
    else:
        raw_values = values

    if not isinstance(raw_values, list):
        raise TypeError("Expected list-like values that can be converted to a Python list.")

    return [float(value) for value in raw_values]


def _validate_histogram(hist: list[float], label: str) -> None:
    """Validate histogram constraints required by the spec."""
    if not hist:
        raise ValueError(f"{label} must not be empty.")

    if any(value < 0.0 for value in hist):
        raise ValueError(f"{label} must not contain negative values.")

    if not isclose(sum(hist), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{label} must sum to 1 within tolerance.")


@dataclass(slots=True)
class AtmosphereState:
    """Atmospheric state used for color signature estimation."""

    cloud_fraction: float
    aerosol_optical_depth: float
    total_ozone_du: float
    humidity: float | None = None
    visibility_km: float | None = None
    pressure_hpa: float | None = None
    cloud_optical_depth: float | None = None
    missing_realtime: bool = False


class SurfaceClass(StrEnum):
    """Basic surface taxonomy for MVP-level surface state."""

    OCEAN = "ocean"
    LAND = "land"
    URBAN = "urban"
    SNOW = "snow"
    DESERT = "desert"
    FOREST = "forest"


@dataclass(slots=True)
class SurfaceState:
    """Surface characteristics used by the signature kernel."""

    surface_class: SurfaceClass
    dominant_albedo: float
    landcover_mix: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ColorSignature:
    """Color signature contract containing sky/ground histograms and metadata."""

    hue_bins: list[float]
    sky_hue_hist: list[float]
    ground_hue_hist: list[float]
    signature: list[float]
    meta: dict[str, Any]
    uncertainty_score: float
    quality_flags: list[str]

    def __post_init__(self) -> None:
        """Validate signature constraints defined in the specification."""
        self.hue_bins = _to_float_list(self.hue_bins)
        self.sky_hue_hist = _to_float_list(self.sky_hue_hist)
        self.ground_hue_hist = _to_float_list(self.ground_hue_hist)
        self.signature = _to_float_list(self.signature)

        _validate_histogram(self.sky_hue_hist, "sky_hue_hist")
        _validate_histogram(self.ground_hue_hist, "ground_hue_hist")

        bins_count = len(self.hue_bins)
        if len(self.sky_hue_hist) != bins_count or len(self.ground_hue_hist) != bins_count:
            raise ValueError("Histogram lengths must match hue_bins length.")

        if len(self.signature) != 2 * bins_count:
            raise ValueError("signature length must equal 2 * len(hue_bins).")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the color signature to a JSON-compatible dictionary."""
        return {
            "hue_bins": self.hue_bins,
            "sky_hue_hist": self.sky_hue_hist,
            "ground_hue_hist": self.ground_hue_hist,
            "signature": self.signature,
            "meta": self.meta,
            "uncertainty_score": self.uncertainty_score,
            "quality_flags": self.quality_flags,
        }
