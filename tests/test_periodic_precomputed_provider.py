"""Tests for precomputed periodic constants provider and surface enrichment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from skycolor_locator.contracts import PeriodicSurfaceConstants, SurfaceClass, SurfaceState
from skycolor_locator.ingest.periodic_precomputed_provider import (
    PrecomputedPeriodicConstantsProvider,
)
from skycolor_locator.ingest.surface_enrichment import merge_surface_with_periodic


def test_precomputed_periodic_provider_returns_matching_tile_period(tmp_path: Path) -> None:
    """Provider should resolve constants for matching tile and time window."""
    path = tmp_path / "periodic.json"
    path.write_text(
        json.dumps(
            [
                {
                    "tile_id": "step0.0500:lat37.5500:lon126.9500",
                    "period_start_utc": "2024-05-01T00:00:00+00:00",
                    "period_end_utc": "2024-05-31T23:59:59+00:00",
                    "landcover_mix": {"urban": 0.7, "land": 0.3},
                    "class_rgb": {"urban": [0.8, 0.75, 0.72]},
                    "meta": {"source": "s2_dynamic_world"},
                }
            ]
        )
    )

    provider = PrecomputedPeriodicConstantsProvider(str(path), tile_step_deg=0.05)
    constants = provider.get_periodic_surface_constants(
        datetime(2024, 5, 12, 9, tzinfo=timezone.utc),
        37.566,
        126.978,
    )

    assert constants.meta["source"] == "s2_dynamic_world"
    assert constants.class_rgb["urban"][0] == pytest.approx(0.8)


def test_merge_surface_with_periodic_overrides_class_rgb_and_mix() -> None:
    """Periodic constants should override/augment surface palette and mix."""
    surface = SurfaceState(
        surface_class=SurfaceClass.URBAN,
        dominant_albedo=0.2,
        landcover_mix={"urban": 1.0},
        class_rgb={"urban": [0.4, 0.4, 0.4]},
    )
    constants = PeriodicSurfaceConstants(
        tile_id="tile",
        period_start_utc=datetime(2024, 5, 1, tzinfo=timezone.utc),
        period_end_utc=datetime(2024, 5, 31, tzinfo=timezone.utc),
        landcover_mix={"urban": 0.6, "land": 0.4},
        class_rgb={"urban": [0.9, 0.8, 0.7]},
        meta={"source": "s2_dynamic_world"},
    )

    merged = merge_surface_with_periodic(surface, constants)

    assert merged.landcover_mix["land"] == pytest.approx(0.4)
    assert merged.class_rgb["urban"][0] == pytest.approx(0.9)
    assert merged.periodic_meta["source"] == "s2_dynamic_world"


def test_precomputed_periodic_provider_fallback_when_missing(tmp_path: Path) -> None:
    """Provider should return empty constants with metadata if no row matches."""
    path = tmp_path / "periodic.json"
    path.write_text(json.dumps([]))

    provider = PrecomputedPeriodicConstantsProvider(str(path), tile_step_deg=0.05)
    constants = provider.get_periodic_surface_constants(
        datetime(2024, 5, 12, 9, tzinfo=timezone.utc),
        37.566,
        126.978,
    )

    assert constants.landcover_mix == {}
    assert constants.class_rgb == {}
    assert constants.meta["source"] == "precomputed_missing"
