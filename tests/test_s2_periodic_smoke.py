"""Smoke tests for opt-in S2 periodic constants builder module."""

from __future__ import annotations

from skycolor_locator.ingest.s2_periodic import S2PeriodicConfig, S2PeriodicConstantsBuilder, tile_id_for


def test_s2_periodic_module_import_and_tile_id_deterministic() -> None:
    """Builder module should import without requiring earthengine-api on default test paths."""
    cfg = S2PeriodicConfig(tile_step_deg=0.05)
    builder = S2PeriodicConstantsBuilder(cfg=cfg)

    assert builder is not None
    assert tile_id_for(37.5665, 126.9780, 0.05) == tile_id_for(37.5665, 126.9780, 0.05)
