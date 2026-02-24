"""Tests for local EarthState batch ingestor."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from skycolor_locator.orchestrate.earthstate_ingest import IngestConfig, ingest_earthstate


def test_ingest_earthstate_mock_writes_expected_rows(tmp_path) -> None:
    """Mock ingest should write deterministic bucket x tile records with upsert semantics."""
    db_path = tmp_path / "earthstate_ingest.db"
    cfg = IngestConfig(
        lat_min=37.5,
        lat_max=37.6,
        lon_min=126.9,
        lon_max=126.9,
        tile_step_deg=0.1,
        start_utc=datetime(2024, 5, 12, 9, 0, tzinfo=UTC),
        end_utc=datetime(2024, 5, 12, 11, 0, tzinfo=UTC),
        bucket_minutes=60,
        provider_mode="mock",
        store_path=str(db_path),
    )

    stats = ingest_earthstate(cfg)
    assert stats["buckets_written"] == 2
    assert stats["tiles_written"] == 4

    with sqlite3.connect(str(db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM earthstate").fetchone()[0]
    assert count == 4

    stats_second = ingest_earthstate(cfg)
    assert stats_second["tiles_written"] == 4
    with sqlite3.connect(str(db_path)) as conn:
        count_after = conn.execute("SELECT COUNT(*) FROM earthstate").fetchone()[0]
    assert count_after == 4
