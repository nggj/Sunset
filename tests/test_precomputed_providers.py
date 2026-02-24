"""Tests for file-backed precomputed ingest providers."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pytest

from skycolor_locator.ingest.precomputed_providers import (
    PrecomputedEarthStateProvider,
    PrecomputedSurfaceProvider,
)


def test_precomputed_earth_state_provider_reads_bucketed_rows(tmp_path: Path) -> None:
    """Provider should return matching atmosphere state for bucketed point/time queries."""
    path = tmp_path / "earth.json"
    path.write_text(
        json.dumps(
            [
                {
                    "time_bucket_utc": "2024-05-12T09:00:00+00:00",
                    "lat": 37.566,
                    "lon": 126.978,
                    "cloud_fraction": 0.4,
                    "aerosol_optical_depth": 0.15,
                    "total_ozone_du": 305.0,
                    "missing_realtime": False,
                }
            ]
        )
    )

    provider = PrecomputedEarthStateProvider(str(path), bucket_minutes=60)
    state = provider.get_atmosphere_state(
        datetime(2024, 5, 12, 9, 45, tzinfo=timezone.utc), 37.566, 126.978
    )

    assert state.cloud_fraction == pytest.approx(0.4)
    assert state.aerosol_optical_depth == pytest.approx(0.15)
    assert state.total_ozone_du == pytest.approx(305.0)


def test_precomputed_surface_provider_reads_rows(tmp_path: Path) -> None:
    """Provider should return matching surface state for quantized point key."""
    path = tmp_path / "surface.json"
    path.write_text(
        json.dumps(
            [
                {
                    "lat": 37.566,
                    "lon": 126.978,
                    "surface_class": "urban",
                    "dominant_albedo": 0.22,
                    "landcover_mix": {"urban": 1.0},
                    "class_rgb": {},
                    "periodic_meta": {"source": "batch"},
                }
            ]
        )
    )

    provider = PrecomputedSurfaceProvider(str(path))
    state = provider.get_surface_state(37.566, 126.978)

    assert state.surface_class.value == "urban"
    assert state.dominant_albedo == pytest.approx(0.22)
    assert state.periodic_meta["source"] == "batch"


def test_precomputed_provider_raises_on_missing_key(tmp_path: Path) -> None:
    """Provider should fail clearly when requested key is not present in snapshots."""
    earth_path = tmp_path / "earth.json"
    earth_path.write_text(
        json.dumps(
            [
                {
                    "time_bucket_utc": "2024-05-12T09:00:00+00:00",
                    "lat": 0.0,
                    "lon": 0.0,
                    "cloud_fraction": 0.4,
                    "aerosol_optical_depth": 0.15,
                    "total_ozone_du": 305.0,
                }
            ]
        )
    )

    provider = PrecomputedEarthStateProvider(str(earth_path), bucket_minutes=60)
    with pytest.raises(KeyError, match="Missing precomputed EarthState"):
        provider.get_atmosphere_state(datetime(2024, 5, 12, 10, tzinfo=timezone.utc), 1.0, 1.0)
