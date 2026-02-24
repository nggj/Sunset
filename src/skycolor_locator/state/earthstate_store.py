"""SQLite-backed local EarthState feature store."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Protocol

from skycolor_locator.contracts import AtmosphereState
from skycolor_locator.time.bucketing import to_utc


@dataclass(frozen=True, slots=True)
class EarthStateKey:
    """Composite deterministic key for EarthState records."""

    bucket_start_utc: datetime
    bucket_minutes: int
    tile_id: str


@dataclass(slots=True)
class EarthStateRecord:
    """One EarthState payload and metadata for a bucket/tile key."""

    key: EarthStateKey
    lat_center: float
    lon_center: float
    atmos: AtmosphereState
    meta: dict[str, Any] = field(default_factory=dict)
    ingested_at_utc: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record to a deterministic JSON-compatible dictionary."""
        return {
            "key": {
                "bucket_start_utc": to_utc(self.key.bucket_start_utc).isoformat(),
                "bucket_minutes": int(self.key.bucket_minutes),
                "tile_id": self.key.tile_id,
            },
            "lat_center": float(self.lat_center),
            "lon_center": float(self.lon_center),
            "atmos": _serialize_atmosphere_state(self.atmos),
            "meta": dict(self.meta),
            "ingested_at_utc": to_utc(self.ingested_at_utc).isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EarthStateRecord":
        """Deserialize a record from a dictionary produced by :meth:`to_dict`."""
        key_raw = data["key"]
        return cls(
            key=EarthStateKey(
                bucket_start_utc=to_utc(datetime.fromisoformat(str(key_raw["bucket_start_utc"]))),
                bucket_minutes=int(key_raw["bucket_minutes"]),
                tile_id=str(key_raw["tile_id"]),
            ),
            lat_center=float(data["lat_center"]),
            lon_center=float(data["lon_center"]),
            atmos=_deserialize_atmosphere_state(data["atmos"]),
            meta=dict(data.get("meta", {})),
            ingested_at_utc=to_utc(datetime.fromisoformat(str(data["ingested_at_utc"]))),
        )


class EarthStateStore(Protocol):
    """Interface for EarthState key-value persistence."""

    def get(self, key: EarthStateKey) -> EarthStateRecord | None:
        """Get one record by key, returning None when not present."""

    def put(self, record: EarthStateRecord) -> None:
        """Insert or replace one record in storage."""

    def bulk_put(self, records: list[EarthStateRecord]) -> None:
        """Insert or replace many records in one operation."""


class SQLiteEarthStateStore(EarthStateStore):
    """Thread-safe SQLite EarthState store using deterministic JSON payload serialization."""

    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS earthstate (
                    bucket_start_utc TEXT NOT NULL,
                    bucket_minutes INTEGER NOT NULL,
                    tile_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (bucket_start_utc, bucket_minutes, tile_id)
                )
                """
            )
            self._conn.commit()

    def get(self, key: EarthStateKey) -> EarthStateRecord | None:
        """Get one record by key, returning None when absent."""
        bucket_start_utc = to_utc(key.bucket_start_utc).isoformat()
        with self._lock:
            row = self._conn.execute(
                """
                SELECT payload_json
                FROM earthstate
                WHERE bucket_start_utc = ? AND bucket_minutes = ? AND tile_id = ?
                """,
                (bucket_start_utc, int(key.bucket_minutes), key.tile_id),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row[0]))
        return EarthStateRecord.from_dict(payload)

    def put(self, record: EarthStateRecord) -> None:
        """Insert or replace one record."""
        payload = record.to_dict()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        key = record.key
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO earthstate (bucket_start_utc, bucket_minutes, tile_id, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    to_utc(key.bucket_start_utc).isoformat(),
                    int(key.bucket_minutes),
                    key.tile_id,
                    payload_json,
                ),
            )
            self._conn.commit()

    def bulk_put(self, records: list[EarthStateRecord]) -> None:
        """Insert or replace many records in a single transaction."""
        entries: list[tuple[str, int, str, str]] = []
        for record in records:
            payload = record.to_dict()
            payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            entries.append(
                (
                    to_utc(record.key.bucket_start_utc).isoformat(),
                    int(record.key.bucket_minutes),
                    record.key.tile_id,
                    payload_json,
                )
            )

        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO earthstate (bucket_start_utc, bucket_minutes, tile_id, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                entries,
            )
            self._conn.commit()


def _serialize_atmosphere_state(atmos: AtmosphereState) -> dict[str, Any]:
    """Serialize :class:`AtmosphereState` using explicit stable field keys."""
    return {
        "cloud_fraction": float(atmos.cloud_fraction),
        "aerosol_optical_depth": float(atmos.aerosol_optical_depth),
        "total_ozone_du": float(atmos.total_ozone_du),
        "humidity": None if atmos.humidity is None else float(atmos.humidity),
        "visibility_km": None if atmos.visibility_km is None else float(atmos.visibility_km),
        "pressure_hpa": None if atmos.pressure_hpa is None else float(atmos.pressure_hpa),
        "cloud_optical_depth": None
        if atmos.cloud_optical_depth is None
        else float(atmos.cloud_optical_depth),
        "cloud_fraction_low": None
        if atmos.cloud_fraction_low is None
        else float(atmos.cloud_fraction_low),
        "cloud_fraction_mid": None
        if atmos.cloud_fraction_mid is None
        else float(atmos.cloud_fraction_mid),
        "cloud_fraction_high": None
        if atmos.cloud_fraction_high is None
        else float(atmos.cloud_fraction_high),
        "cloud_optical_depth_low": None
        if atmos.cloud_optical_depth_low is None
        else float(atmos.cloud_optical_depth_low),
        "cloud_optical_depth_mid": None
        if atmos.cloud_optical_depth_mid is None
        else float(atmos.cloud_optical_depth_mid),
        "cloud_optical_depth_high": None
        if atmos.cloud_optical_depth_high is None
        else float(atmos.cloud_optical_depth_high),
        "cloud_fraction_sat": None
        if atmos.cloud_fraction_sat is None
        else float(atmos.cloud_fraction_sat),
        "cloud_optical_depth_sat": None
        if atmos.cloud_optical_depth_sat is None
        else float(atmos.cloud_optical_depth_sat),
        "cloud_top_height_m": None
        if atmos.cloud_top_height_m is None
        else float(atmos.cloud_top_height_m),
        "cloud_base_height_m": None
        if atmos.cloud_base_height_m is None
        else float(atmos.cloud_base_height_m),
        "cloud_top_pressure_hpa": None
        if atmos.cloud_top_pressure_hpa is None
        else float(atmos.cloud_top_pressure_hpa),
        "cloud_base_pressure_hpa": None
        if atmos.cloud_base_pressure_hpa is None
        else float(atmos.cloud_base_pressure_hpa),
        "missing_realtime": bool(atmos.missing_realtime),
    }


def _deserialize_atmosphere_state(data: dict[str, Any]) -> AtmosphereState:
    """Deserialize :class:`AtmosphereState` from dictionary representation."""
    return AtmosphereState(
        cloud_fraction=float(data["cloud_fraction"]),
        aerosol_optical_depth=float(data["aerosol_optical_depth"]),
        total_ozone_du=float(data["total_ozone_du"]),
        humidity=None if data.get("humidity") is None else float(data["humidity"]),
        visibility_km=None if data.get("visibility_km") is None else float(data["visibility_km"]),
        pressure_hpa=None if data.get("pressure_hpa") is None else float(data["pressure_hpa"]),
        cloud_optical_depth=None
        if data.get("cloud_optical_depth") is None
        else float(data["cloud_optical_depth"]),
        cloud_fraction_low=None
        if data.get("cloud_fraction_low") is None
        else float(data["cloud_fraction_low"]),
        cloud_fraction_mid=None
        if data.get("cloud_fraction_mid") is None
        else float(data["cloud_fraction_mid"]),
        cloud_fraction_high=None
        if data.get("cloud_fraction_high") is None
        else float(data["cloud_fraction_high"]),
        cloud_optical_depth_low=None
        if data.get("cloud_optical_depth_low") is None
        else float(data["cloud_optical_depth_low"]),
        cloud_optical_depth_mid=None
        if data.get("cloud_optical_depth_mid") is None
        else float(data["cloud_optical_depth_mid"]),
        cloud_optical_depth_high=None
        if data.get("cloud_optical_depth_high") is None
        else float(data["cloud_optical_depth_high"]),
        cloud_fraction_sat=None
        if data.get("cloud_fraction_sat") is None
        else float(data["cloud_fraction_sat"]),
        cloud_optical_depth_sat=None
        if data.get("cloud_optical_depth_sat") is None
        else float(data["cloud_optical_depth_sat"]),
        cloud_top_height_m=None
        if data.get("cloud_top_height_m") is None
        else float(data["cloud_top_height_m"]),
        cloud_base_height_m=None
        if data.get("cloud_base_height_m") is None
        else float(data["cloud_base_height_m"]),
        cloud_top_pressure_hpa=None
        if data.get("cloud_top_pressure_hpa") is None
        else float(data["cloud_top_pressure_hpa"]),
        cloud_base_pressure_hpa=None
        if data.get("cloud_base_pressure_hpa") is None
        else float(data["cloud_base_pressure_hpa"]),
        missing_realtime=bool(data.get("missing_realtime", False)),
    )
