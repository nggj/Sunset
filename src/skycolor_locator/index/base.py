"""Base interface for vector indices."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

NDArray: TypeAlias = Any


class VectorIndex(Protocol):
    """Protocol for vector index implementations."""

    def add(self, keys: list[str], vectors: NDArray) -> None:
        """Add vectors identified by string keys to the index."""

    def query(self, vector: NDArray, top_k: int) -> list[tuple[str, float]]:
        """Query nearest vectors and return (key, distance) pairs."""
