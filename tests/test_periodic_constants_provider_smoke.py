"""Smoke tests for periodic constants provider interfaces and mocks."""

from __future__ import annotations

from datetime import datetime, timezone

from skycolor_locator.ingest.interfaces import PeriodicConstantsProvider, SurfaceProvider
from skycolor_locator.ingest.mock_providers import MockPeriodicConstantsProvider, MockSurfaceProvider


def test_periodic_constants_provider_import_and_runtime_checkable_usage() -> None:
    """Protocol and mock class should import and interoperate without breaking SurfaceProvider."""
    surface_provider: SurfaceProvider = MockSurfaceProvider()
    periodic_provider: PeriodicConstantsProvider = MockPeriodicConstantsProvider()

    surface = surface_provider.get_surface_state(37.0, 127.0)
    constants = periodic_provider.get_periodic_surface_constants(
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        37.0,
        127.0,
    )

    assert surface.surface_class.value in {"ocean", "land", "urban", "snow", "desert", "forest"}
    assert constants.landcover_mix == {}
    assert constants.class_rgb == {}
    assert constants.meta["source"] == "mock"
