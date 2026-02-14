"""Base interface for vector indices."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

NDArray: TypeAlias = Any


class VectorIndex(Protocol):
    """Protocol for vector index implementations."""

    def add(self, keys: list[str], vectors: NDArray, metadatas: list[dict[str, Any]] | None = None) -> None:
        """Add vectors and optional metadata identified by string keys to the index."""

    def query(
        self,
        vector: NDArray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Query nearest vectors with optional filters and return (key, score) pairs."""
