"""Tests for circular Earth Mover's Distance helpers."""

from __future__ import annotations

from skycolor_locator.index.metrics import circular_emd_1d, emd_1d


def _one_hot(length: int, index: int) -> list[float]:
    """Return a one-hot histogram of the requested length."""
    values = [0.0] * length
    values[index] = 1.0
    return values


def test_circular_emd_wraparound_is_small_vs_linear() -> None:
    """Wrap-around bins should be close in circular EMD but far in linear EMD."""
    bins = 12
    a = _one_hot(bins, 0)
    b = _one_hot(bins, bins - 1)

    assert emd_1d(a, b) == bins - 1
    assert circular_emd_1d(a, b) == 1.0


def test_circular_emd_symmetry() -> None:
    """Circular EMD should be symmetric."""
    a = [0.4, 0.3, 0.2, 0.1]
    b = [0.1, 0.2, 0.3, 0.4]

    assert circular_emd_1d(a, b) == circular_emd_1d(b, a)


def test_circular_emd_identity() -> None:
    """Circular EMD of identical histograms should be zero."""
    a = [0.25, 0.25, 0.25, 0.25]

    assert circular_emd_1d(a, a) == 0.0
