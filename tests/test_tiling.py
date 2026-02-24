"""Tests for deterministic geospatial tiling helpers."""

from skycolor_locator.geo.tiling import tile_id_for


def test_tile_id_for_known_values() -> None:
    """tile_id_for should preserve stable string formatting and tile origin values."""
    assert tile_id_for(37.5665, 126.9780, 0.05) == "step0.0500:lat37.5500:lon126.9500"
    assert tile_id_for(-33.8688, 151.2093, 0.1) == "step0.1000:lat-33.9000:lon151.2000"
    assert tile_id_for(0.0, -0.0001, 0.25) == "step0.2500:lat0.0000:lon-0.2500"
