"""In-memory brute-force vector index for MVP usage."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypeAlias

from skycolor_locator.index.metrics import cosine_distance

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


def _parse_time(value: Any) -> datetime | None:
    """Parse datetime-like metadata or filter values."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            if value.endswith("Z"):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _landcover_match(metadata: dict[str, Any], filter_value: Any) -> bool:
    """Check whether metadata satisfies a landcover filter."""
    tags = metadata.get("landcover_tags")
    if tags is None:
        return False

    if isinstance(filter_value, str):
        wanted = {filter_value}
    elif isinstance(filter_value, list):
        wanted = {str(v) for v in filter_value}
    else:
        return False

    if isinstance(tags, str):
        present = {tags}
    elif isinstance(tags, list):
        present = {str(v) for v in tags}
    else:
        return False

    return bool(wanted & present)


def _metadata_matches(metadata: dict[str, Any] | None, filters: dict[str, Any] | None) -> bool:
    """Evaluate basic metadata filters for time and landcover."""
    if not filters:
        return True
    if metadata is None:
        return False

    time_min = _parse_time(filters.get("time_min")) if "time_min" in filters else None
    time_max = _parse_time(filters.get("time_max")) if "time_max" in filters else None
    if time_min is not None or time_max is not None:
        meta_time = _parse_time(metadata.get("time_utc"))
        if meta_time is None:
            return False
        if time_min is not None and meta_time < time_min:
            return False
        if time_max is not None and meta_time > time_max:
            return False

    if "landcover" in filters and not _landcover_match(metadata, filters["landcover"]):
        return False

    return True


class BruteforceIndex:
    """Brute-force vector index supporting cosine or dot-product scoring."""

    def __init__(self, mode: Literal["cosine", "dot"] = "cosine") -> None:
        """Initialize index.

        Args:
            mode: `"cosine"` for cosine distance, `"dot"` for negative inner-product distance.
        """
        if mode not in {"cosine", "dot"}:
            raise ValueError("mode must be 'cosine' or 'dot'")
        self.mode = mode
        self._keys: list[str] = []
        self._vectors: list[list[float]] = []
        self._metadatas: list[dict[str, Any] | None] = []

    def add(self, keys: list[str], vectors: NDArray, metadatas: list[dict[str, Any]] | None = None) -> None:
        """Add vectors with keys and optional metadata into in-memory storage."""
        vecs = _to_vector_list(vectors)
        if len(keys) != len(vecs):
            raise ValueError("keys and vectors must have the same length")
        if metadatas is not None and len(metadatas) != len(keys):
            raise ValueError("metadatas and keys must have the same length")

        if self._vectors and vecs and len(self._vectors[0]) != len(vecs[0]):
            raise ValueError("all vectors must have identical dimensions")

        self._keys.extend(keys)
        self._vectors.extend(vecs)
        if metadatas is None:
            self._metadatas.extend([None] * len(keys))
        else:
            self._metadatas.extend(metadatas)

    def query(
        self,
        vector: NDArray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Return nearest keys by configured metric and optional metadata filters."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._vectors:
            return []

        q = _to_query_vector(vector)
        if len(q) != len(self._vectors[0]):
            raise ValueError("query vector dimension mismatch")

        scored: list[tuple[str, float]] = []
        for key, candidate, metadata in zip(self._keys, self._vectors, self._metadatas, strict=True):
            if not _metadata_matches(metadata, filters):
                continue
            if self.mode == "cosine":
                dist = cosine_distance(q, candidate)
            else:
                dist = -sum(x * y for x, y in zip(q, candidate, strict=True))
            scored.append((key, dist))

        scored.sort(key=lambda item: item[1])
        return scored[: min(top_k, len(scored))]
