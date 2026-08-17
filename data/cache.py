"""Thread-safe LRU cache for candles / features / orderbook snapshots."""
from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Hashable, Optional


class LRUCache:
    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._store: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def invalidate(self, key: Hashable) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._store
