"""A small TTL + LRU cache used for query-plan caching (docs/06-query-understanding.md §3)."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """Thread-safe LRU cache whose entries expire ``ttl_s`` seconds after insertion."""

    def __init__(
        self,
        maxsize: int = 1024,
        ttl_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._clock = clock
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> V | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            expires, value = item
            if expires <= self._clock():
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = (self._clock() + self.ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
