"""Tests for local SQLite EarthState feature store."""

from datetime import UTC, datetime

from skycolor_locator.contracts import AtmosphereState
from skycolor_locator.state.earthstate_store import (
    EarthStateKey,
    EarthStateRecord,
    SQLiteEarthStateStore,
)


def _sample_record(tile_id: str = "step0.0500:lat37.5500:lon126.9500") -> EarthStateRecord:
    return EarthStateRecord(
        key=EarthStateKey(
            bucket_start_utc=datetime(2024, 5, 12, 9, 0, tzinfo=UTC),
            bucket_minutes=60,
            tile_id=tile_id,
        ),
        lat_center=37.575,
        lon_center=126.975,
        atmos=AtmosphereState(
            cloud_fraction=0.21,
            aerosol_optical_depth=0.15,
            total_ozone_du=301.3,
            humidity=55.0,
            visibility_km=16.2,
            pressure_hpa=1012.4,
            cloud_optical_depth=4.1,
            missing_realtime=False,
        ),
        meta={"provider": "mock", "source_time": "2024-05-12T09:03:00+00:00"},
        ingested_at_utc=datetime(2024, 5, 12, 9, 5, tzinfo=UTC),
    )


def test_sqlite_earthstate_store_roundtrip(tmp_path) -> None:
    """Store should persist and load one EarthState record without value drift."""
    db = tmp_path / "earthstate.db"
    store = SQLiteEarthStateStore(str(db))
    record = _sample_record()

    store.put(record)
    loaded = store.get(record.key)

    assert loaded is not None
    assert loaded.key == record.key
    assert loaded.lat_center == record.lat_center
    assert loaded.lon_center == record.lon_center
    assert loaded.meta == record.meta
    assert loaded.ingested_at_utc == record.ingested_at_utc

    assert loaded.atmos.cloud_fraction == record.atmos.cloud_fraction
    assert loaded.atmos.aerosol_optical_depth == record.atmos.aerosol_optical_depth
    assert loaded.atmos.total_ozone_du == record.atmos.total_ozone_du
    assert loaded.atmos.humidity == record.atmos.humidity
    assert loaded.atmos.visibility_km == record.atmos.visibility_km
    assert loaded.atmos.pressure_hpa == record.atmos.pressure_hpa
    assert loaded.atmos.cloud_optical_depth == record.atmos.cloud_optical_depth
    assert loaded.atmos.missing_realtime == record.atmos.missing_realtime


def test_sqlite_earthstate_store_bulk_put(tmp_path) -> None:
    """Store should insert multiple rows via bulk_put."""
    db = tmp_path / "earthstate.db"
    store = SQLiteEarthStateStore(str(db))
    rec1 = _sample_record("step0.0500:lat37.5500:lon126.9500")
    rec2 = _sample_record("step0.0500:lat37.6000:lon126.9500")

    store.bulk_put([rec1, rec2])

    assert store.get(rec1.key) is not None
    assert store.get(rec2.key) is not None


def test_sqlite_earthstate_store_missing_returns_none(tmp_path) -> None:
    """Store should return None for a missing key."""
    db = tmp_path / "earthstate.db"
    store = SQLiteEarthStateStore(str(db))

    missing = EarthStateKey(
        bucket_start_utc=datetime(2024, 5, 12, 10, 0, tzinfo=UTC),
        bucket_minutes=60,
        tile_id="step0.0500:lat37.5500:lon126.9500",
    )

    assert store.get(missing) is None
