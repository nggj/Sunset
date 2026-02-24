"""SQLite-backed periodic surface constants store."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Protocol

from skycolor_locator.contracts import PeriodicSurfaceConstants
from skycolor_locator.time.bucketing import to_utc


@dataclass(frozen=True, slots=True)
class PeriodicKey:
    """Key for periodic constants identified by tile and UTC period bounds."""

    tile_id: str
    period_start_utc: datetime
    period_end_utc: datetime


@dataclass(slots=True)
class PeriodicRecord:
    """Periodic constants record persisted in store."""

    key: PeriodicKey
    constants: PeriodicSurfaceConstants
    meta: dict[str, Any] = field(default_factory=dict)
    ingested_at_utc: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to deterministic JSON-compatible dictionary."""
        return {
            "key": {
                "tile_id": self.key.tile_id,
                "period_start_utc": to_utc(self.key.period_start_utc).isoformat(),
                "period_end_utc": to_utc(self.key.period_end_utc).isoformat(),
            },
            "constants": {
                "tile_id": self.constants.tile_id,
                "period_start_utc": to_utc(self.constants.period_start_utc).isoformat(),
                "period_end_utc": to_utc(self.constants.period_end_utc).isoformat(),
                "landcover_mix": dict(self.constants.landcover_mix),
                "class_rgb": {k: list(v) for k, v in self.constants.class_rgb.items()},
                "meta": dict(self.constants.meta),
            },
            "meta": dict(self.meta),
            "ingested_at_utc": to_utc(self.ingested_at_utc).isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PeriodicRecord":
        """Deserialize record from dictionary created by :meth:`to_dict`."""
        key_raw = data["key"]
        constants_raw = data["constants"]
        return cls(
            key=PeriodicKey(
                tile_id=str(key_raw["tile_id"]),
                period_start_utc=to_utc(datetime.fromisoformat(str(key_raw["period_start_utc"]))),
                period_end_utc=to_utc(datetime.fromisoformat(str(key_raw["period_end_utc"]))),
            ),
            constants=PeriodicSurfaceConstants(
                tile_id=str(constants_raw["tile_id"]),
                period_start_utc=to_utc(datetime.fromisoformat(str(constants_raw["period_start_utc"]))),
                period_end_utc=to_utc(datetime.fromisoformat(str(constants_raw["period_end_utc"]))),
                landcover_mix={k: float(v) for k, v in dict(constants_raw["landcover_mix"]).items()},
                class_rgb={
                    str(k): [float(x) for x in list(v)]
                    for k, v in dict(constants_raw["class_rgb"]).items()
                },
                meta=dict(constants_raw.get("meta", {})),
            ),
            meta=dict(data.get("meta", {})),
            ingested_at_utc=to_utc(datetime.fromisoformat(str(data["ingested_at_utc"]))),
        )


class PeriodicConstantsStore(Protocol):
    """Store protocol for periodic constants records."""

    def get(self, key: PeriodicKey) -> PeriodicRecord | None:
        """Get one periodic record by key."""

    def put(self, record: PeriodicRecord) -> None:
        """Insert or replace one periodic record."""


class SQLitePeriodicConstantsStore(PeriodicConstantsStore):
    """Thread-safe SQLite store for periodic constants payloads."""

    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS periodic_constants (
                    tile_id TEXT NOT NULL,
                    period_start_utc TEXT NOT NULL,
                    period_end_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (tile_id, period_start_utc, period_end_utc)
                )
                """
            )
            self._conn.commit()

    def get(self, key: PeriodicKey) -> PeriodicRecord | None:
        """Return periodic record by key or None when missing."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT payload_json
                FROM periodic_constants
                WHERE tile_id = ? AND period_start_utc = ? AND period_end_utc = ?
                """,
                (
                    key.tile_id,
                    to_utc(key.period_start_utc).isoformat(),
                    to_utc(key.period_end_utc).isoformat(),
                ),
            ).fetchone()
        if row is None:
            return None
        return PeriodicRecord.from_dict(json.loads(str(row[0])))

    def put(self, record: PeriodicRecord) -> None:
        """Insert or replace one periodic record."""
        payload_json = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO periodic_constants
                (tile_id, period_start_utc, period_end_utc, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    record.key.tile_id,
                    to_utc(record.key.period_start_utc).isoformat(),
                    to_utc(record.key.period_end_utc).isoformat(),
                    payload_json,
                ),
            )
            self._conn.commit()
