"""Tests for local brute-force vector index."""

from __future__ import annotations

from skycolor_locator.index.bruteforce import BruteforceIndex


def test_bruteforce_index_returns_expected_nearest_vector() -> None:
    """Given three vectors, the nearest neighbor should match expectation."""
    index = BruteforceIndex(mode="cosine")
    keys = ["a", "b", "c"]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    index.add(keys, vectors)

    results = index.query([0.9, 0.1, 0.0], top_k=2)

    assert results[0][0] == "a"
    assert len(results) == 2


def test_bruteforce_index_dot_mode() -> None:
    """Dot mode should prioritize the highest inner-product vector."""
    index = BruteforceIndex(mode="dot")
    index.add(
        ["x", "y", "z"],
        [
            [0.3, 0.2],
            [0.6, 0.4],
            [0.1, 0.9],
        ],
    )

    results = index.query([0.5, 0.4], top_k=1)

    assert results[0][0] == "y"


def test_bruteforce_index_time_filter_restricts_results() -> None:
    """time_min/time_max filters should keep only metadata-matching vectors."""
    index = BruteforceIndex(mode="cosine")
    index.add(
        ["early", "late"],
        [[1.0, 0.0], [1.0, 0.0]],
        metadatas=[
            {"time_utc": "2024-01-01T00:00:00+00:00", "landcover_tags": ["urban"]},
            {"time_utc": "2024-06-01T00:00:00+00:00", "landcover_tags": ["urban"]},
        ],
    )

    results = index.query(
        [1.0, 0.0],
        top_k=2,
        filters={"time_min": "2024-05-01T00:00:00+00:00", "time_max": "2024-12-31T00:00:00+00:00"},
    )

    assert [key for key, _ in results] == ["late"]


def test_bruteforce_index_landcover_filter_restricts_results() -> None:
    """landcover filter should keep only matching tagged vectors."""
    index = BruteforceIndex(mode="cosine")
    index.add(
        ["urban_key", "forest_key"],
        [[0.9, 0.1], [0.9, 0.1]],
        metadatas=[
            {"time_utc": "2024-01-01T00:00:00+00:00", "landcover_tags": ["urban", "land"]},
            {"time_utc": "2024-01-01T00:00:00+00:00", "landcover_tags": ["forest"]},
        ],
    )

    results = index.query([0.9, 0.1], top_k=2, filters={"landcover": "forest"})

    assert [key for key, _ in results] == ["forest_key"]
