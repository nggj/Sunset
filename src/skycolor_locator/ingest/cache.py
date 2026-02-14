"""Small LRU cache utility for provider results."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """A minimal deterministic LRU cache."""

    def __init__(self, capacity: int = 128) -> None:
        """Initialize cache with positive capacity."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K, factory: Callable[[], V]) -> V:
        """Get cached value or create/store via `factory`."""
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]

        value = factory()
        self._items[key] = value
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return value
