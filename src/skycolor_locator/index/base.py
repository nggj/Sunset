"""Base interface for vector indices."""

from __future__ import annotations

from typing import Protocol


class np:
    """Minimal numpy namespace shim for ndarray type annotations."""

    ndarray = list[float]


class VectorIndex(Protocol):
    """Protocol for vector index implementations."""

    def add(self, keys: list[str], vectors: np.ndarray) -> None:
        """Add vectors identified by string keys to the index."""

    def query(self, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Query nearest vectors and return (key, distance) pairs."""
