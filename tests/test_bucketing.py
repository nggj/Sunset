"""Tests for deterministic UTC time bucketing helpers."""

from datetime import UTC, datetime, timedelta, timezone

from skycolor_locator.time.bucketing import bucket_start_utc


def test_bucket_start_utc_with_naive_datetime() -> None:
    """Naive datetimes should be treated as UTC and floored to bucket starts."""
    dt = datetime(2024, 5, 12, 9, 37, 45, 123456)
    assert bucket_start_utc(dt, 60) == datetime(2024, 5, 12, 9, 0, tzinfo=UTC)


def test_bucket_start_utc_with_timezone_aware_datetime() -> None:
    """Aware datetimes should be converted to UTC before bucketing."""
    dt = datetime(2024, 5, 12, 18, 5, 30, tzinfo=timezone.utc)
    assert bucket_start_utc(dt, 15) == datetime(2024, 5, 12, 18, 0, tzinfo=UTC)


def test_bucket_start_utc_with_non_utc_timezone() -> None:
    """Non-UTC aware datetimes should normalize and bucket in UTC."""
    kst = timezone(timedelta(hours=9))
    dt = datetime(2024, 5, 13, 3, 5, 30, tzinfo=kst)
    assert bucket_start_utc(dt, 30) == datetime(2024, 5, 12, 18, 0, tzinfo=UTC)
