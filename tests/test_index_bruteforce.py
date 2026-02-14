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
