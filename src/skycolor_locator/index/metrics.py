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
