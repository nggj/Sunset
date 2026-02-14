"""Tests for brute-force index EMD modes."""

from __future__ import annotations

from skycolor_locator.index.bruteforce import BruteforceIndex


def test_circular_emd_mode_prefers_wraparound_neighbor() -> None:
    """Circular EMD mode should rank wrap-adjacent hue as nearest."""
    # Signature layout: [sky bins..., ground bins...]
    # Use four bins per half for compact wrap-around behavior.
    query = [1.0, 0.0, 0.0, 0.0] + [1.0, 0.0, 0.0, 0.0]
    wrap_neighbor = [0.0, 0.0, 0.0, 1.0] + [0.0, 0.0, 0.0, 1.0]
    opposite = [0.0, 0.0, 1.0, 0.0] + [0.0, 0.0, 1.0, 0.0]

    index = BruteforceIndex(mode="circular_emd")
    index.add(["wrap", "opposite"], [wrap_neighbor, opposite])

    results = index.query(query, top_k=2)

    assert results[0][0] == "wrap"
    assert results[0][1] < results[1][1]
