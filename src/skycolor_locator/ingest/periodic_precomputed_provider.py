"""Precomputed periodic constants provider for runtime surface enrichment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from skycolor_locator.contracts import PeriodicSurfaceConstants
from skycolor_locator.ingest.interfaces import PeriodicConstantsProvider
from skycolor_locator.ingest.s2_periodic import tile_id_for


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return {str(k): v for k, v in value.items()}


def _as_float(value: object, field: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{field} must be numeric")


def _as_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be ISO datetime string")
    dt = datetime.fromisoformat(value)
    return dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _as_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return value


class PrecomputedPeriodicConstantsProvider(PeriodicConstantsProvider):
    """Load periodic constants snapshots and resolve tile/period matches deterministically."""

    def __init__(self, dataset_path: str, tile_step_deg: float = 0.05) -> None:
        self._tile_step_deg = tile_step_deg
        payload = json.loads(Path(dataset_path).read_text())
        rows: list[object]
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            rows = payload["rows"]
        else:
            raise ValueError(f"Unsupported periodic dataset format: {dataset_path}")

        self._by_tile: dict[str, list[PeriodicSurfaceConstants]] = {}
        for row in rows:
            mapped = _as_mapping(row, "periodic_row")
            constants = PeriodicSurfaceConstants(
                tile_id=str(mapped["tile_id"]),
                period_start_utc=_as_datetime(mapped["period_start_utc"], "period_start_utc"),
                period_end_utc=_as_datetime(mapped["period_end_utc"], "period_end_utc"),
                landcover_mix={
                    str(k): _as_float(v, "landcover_mix value")
                    for k, v in _as_mapping(mapped.get("landcover_mix", {}), "landcover_mix").items()
                },
                class_rgb={
                    str(k): [_as_float(c, "class_rgb channel") for c in _as_list(v, "class_rgb")]
                    for k, v in _as_mapping(mapped.get("class_rgb", {}), "class_rgb").items()
                },
                meta=dict(_as_mapping(mapped.get("meta", {}), "meta")),
            )
            self._by_tile.setdefault(constants.tile_id, []).append(constants)

        for tile in self._by_tile:
            self._by_tile[tile].sort(key=lambda item: item.period_start_utc)

    def get_periodic_surface_constants(
        self, dt: datetime, lat: float, lon: float
    ) -> PeriodicSurfaceConstants:
        """Return constants whose tile matches and period contains dt, else empty constants."""
        dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        tile_id = tile_id_for(lat, lon, self._tile_step_deg)
        candidates = self._by_tile.get(tile_id, [])

        selected: PeriodicSurfaceConstants | None = None
        for item in candidates:
            if item.period_start_utc <= dt_utc <= item.period_end_utc:
                selected = item
        if selected is not None:
            return selected

        return PeriodicSurfaceConstants(
            tile_id=tile_id,
            period_start_utc=dt_utc,
            period_end_utc=dt_utc,
            landcover_mix={},
            class_rgb={},
            meta={"source": "precomputed_missing"},
        )
