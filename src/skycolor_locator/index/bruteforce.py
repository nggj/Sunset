"""In-memory brute-force vector index for MVP usage."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from skycolor_locator.index.metrics import (
    circular_emd_signature_halves,
    cosine_distance,
    emd_signature_halves,
)

NDArray: TypeAlias = Any


def _to_vector_list(vectors: NDArray) -> list[list[float]]:
    """Convert numpy-like 2D arrays into Python list vectors."""
    if hasattr(vectors, "tolist"):
        raw = vectors.tolist()
    else:
        raw = vectors

    if not isinstance(raw, list):
        raise TypeError("vectors must be a list-like 2D array")

    converted: list[list[float]] = []
    for row in raw:
        if not isinstance(row, list):
            raise TypeError("vectors must be a 2D list-like array")
        converted.append([float(value) for value in row])
    return converted


def _to_query_vector(vector: NDArray) -> list[float]:
    """Convert numpy-like 1D array into a Python list."""
    if hasattr(vector, "tolist"):
        raw = vector.tolist()
    else:
        raw = vector

    if not isinstance(raw, list):
        raise TypeError("vector must be a list-like 1D array")
    return [float(value) for value in raw]


class BruteforceIndex:
    """Brute-force vector index supporting cosine, dot-product, and EMD scoring."""

    def __init__(
        self, mode: Literal["cosine", "dot", "emd", "circular_emd"] = "cosine"
    ) -> None:
        """Initialize index.

        Args:
            mode: `"cosine"` for cosine distance, `"dot"` for negative inner-product
                distance, `"emd"` for linear histogram EMD, and
                `"circular_emd"` for circular histogram EMD.
        """
        if mode not in {"cosine", "dot", "emd", "circular_emd"}:
            raise ValueError("mode must be one of: cosine, dot, emd, circular_emd")
        self.mode = mode
        self._keys: list[str] = []
        self._vectors: list[list[float]] = []

    def add(self, keys: list[str], vectors: NDArray) -> None:
        """Add vectors with keys into in-memory storage."""
        vecs = _to_vector_list(vectors)
        if len(keys) != len(vecs):
            raise ValueError("keys and vectors must have the same length")

        if self._vectors and vecs and len(self._vectors[0]) != len(vecs[0]):
            raise ValueError("all vectors must have identical dimensions")

        self._keys.extend(keys)
        self._vectors.extend(vecs)

    def query(self, vector: NDArray, top_k: int) -> list[tuple[str, float]]:
        """Return nearest keys by configured distance metric."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._vectors:
            return []

        q = _to_query_vector(vector)
        if len(q) != len(self._vectors[0]):
            raise ValueError("query vector dimension mismatch")

        scored: list[tuple[str, float]] = []
        for key, candidate in zip(self._keys, self._vectors, strict=True):
            if self.mode == "cosine":
                dist = cosine_distance(q, candidate)
            elif self.mode == "dot":
                dist = -sum(x * y for x, y in zip(q, candidate, strict=True))
            elif self.mode == "emd":
                dist = emd_signature_halves(q, candidate)
            else:
                dist = circular_emd_signature_halves(q, candidate)
            scored.append((key, dist))

        scored.sort(key=lambda item: item[1])
        return scored[: min(top_k, len(scored))]
