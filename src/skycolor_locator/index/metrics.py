"""Distance/similarity metrics for signature vectors."""

from __future__ import annotations

from math import sqrt


def cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance (1 - cosine similarity) for equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def emd_1d(a: list[float], b: list[float]) -> float:
    """Compute 1D Earth Mover's Distance for equal-length histograms."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")

    flow = 0.0
    distance = 0.0
    for x, y in zip(a, b, strict=True):
        flow += x - y
        distance += abs(flow)
    return distance


def median(values: list[float]) -> float:
    """Compute median for a non-empty list of floats in pure Python."""
    if not values:
        raise ValueError("values must be non-empty")

    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def circular_emd_1d(a: list[float], b: list[float]) -> float:
    """Compute circular 1D Earth Mover's Distance for equal-length histograms."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")
    if not a:
        raise ValueError("Vectors must be non-empty.")

    cumsums: list[float] = []
    cumulative = 0.0
    for x, y in zip(a, b, strict=True):
        cumulative += x - y
        cumsums.append(cumulative)

    med = median(cumsums)
    return sum(abs(ci - med) for ci in cumsums)


def emd_signature_halves(sig_a: list[float], sig_b: list[float]) -> float:
    """Compute linear EMD across sky and ground signature halves independently."""
    if len(sig_a) != len(sig_b):
        raise ValueError("Vectors must have the same length.")
    if len(sig_a) % 2 != 0:
        raise ValueError("signature length must be even")

    half = len(sig_a) // 2
    return emd_1d(sig_a[:half], sig_b[:half]) + emd_1d(sig_a[half:], sig_b[half:])


def circular_emd_signature_halves(sig_a: list[float], sig_b: list[float]) -> float:
    """Compute circular EMD across sky and ground signature halves independently."""
    if len(sig_a) != len(sig_b):
        raise ValueError("Vectors must have the same length.")
    if len(sig_a) % 2 != 0:
        raise ValueError("signature length must be even")

    half = len(sig_a) // 2
    return circular_emd_1d(sig_a[:half], sig_b[:half]) + circular_emd_1d(
        sig_a[half:], sig_b[half:]
    )
