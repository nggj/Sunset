"""In-memory cache store for reusable search indices."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Callable

from skycolor_locator.index.bruteforce import BruteforceIndex


@dataclass(frozen=True)
class IndexCacheKey:
    """Stable cache key for one prebuilt candidate index."""

    time_bucket: str
    vector_type: str
    vector_dim: int
    grid_spec_hash: str
    model_version: str
    metric: str
    apply_residual: bool


@dataclass
class IndexEntry:
    """Stored index payload with build metadata."""

    index: BruteforceIndex
    built_at: datetime
    metadata_by_key: dict[str, dict[str, Any]]


class IndexStore:
    """Thread-safe in-memory cache with TTL and LRU-style eviction."""

    def __init__(self, ttl_seconds: int = 600, max_entries: int = 8) -> None:
        """Initialize store with TTL and maximum retained entries."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_entries = max_entries
        self._lock = Lock()
        self._entries: OrderedDict[IndexCacheKey, IndexEntry] = OrderedDict()
        self.build_count = 0

    def _is_expired(self, entry: IndexEntry, now: datetime) -> bool:
        return entry.built_at + self._ttl <= now

    def _purge_expired_locked(self, now: datetime) -> None:
        expired = [key for key, entry in self._entries.items() if self._is_expired(entry, now)]
        for key in expired:
            self._entries.pop(key, None)

    def get_or_build(
        self,
        key: IndexCacheKey,
        builder: Callable[[], tuple[BruteforceIndex, dict[str, dict[str, Any]]]],
    ) -> tuple[IndexEntry, bool]:
        """Return cached entry for key, building once on miss."""
        now = datetime.now(timezone.utc)
        with self._lock:
            self._purge_expired_locked(now)
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing, False

            index, metadata_by_key = builder()
            entry = IndexEntry(index=index, built_at=now, metadata_by_key=metadata_by_key)
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self.build_count += 1

            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

            return entry, True
