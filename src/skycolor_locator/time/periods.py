"""Deterministic UTC period selection helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from skycolor_locator.time.bucketing import to_utc


def period_month(dt_utc: datetime) -> tuple[datetime, datetime]:
    """Return UTC month period bounds as [start, end) for an input datetime."""
    dt = to_utc(dt_utc)
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start.astimezone(UTC), end.astimezone(UTC)
