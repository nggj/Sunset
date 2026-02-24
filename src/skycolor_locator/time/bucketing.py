"""Deterministic UTC time bucketing utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from collections.abc import Iterator


def to_utc(dt: datetime) -> datetime:
    """Convert datetime to timezone-aware UTC, assuming UTC for naive values."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def bucket_start_utc(dt: datetime, bucket_minutes: int) -> datetime:
    """Return UTC bucket start for the provided datetime and bucket size."""
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be positive")
    out = to_utc(dt).replace(second=0, microsecond=0)
    minute = out.minute - (out.minute % bucket_minutes)
    return out.replace(minute=minute)


def iter_bucket_starts(start_utc: datetime, end_utc: datetime, bucket_minutes: int) -> Iterator[datetime]:
    """Yield bucket starts from [start_utc, end_utc) at bucket_minutes cadence in UTC."""
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be positive")
    current = bucket_start_utc(start_utc, bucket_minutes)
    end = to_utc(end_utc)
    step = timedelta(minutes=bucket_minutes)
    while current < end:
        yield current
        current = current + step
