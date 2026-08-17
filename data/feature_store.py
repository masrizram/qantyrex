"""Feature store: caches computed feature frames per (symbol, timeframe).

Acts as the source of truth for "what features were available at time T",
enforcing that features are computed only from closed candles (no look-ahead).
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .cache import LRUCache


class FeatureStore:
    def __init__(self, capacity: int = 64) -> None:
        self._cache = LRUCache(capacity=capacity)

    def _key(self, symbol: str, timeframe: str) -> Tuple[str, str]:
        return (symbol, timeframe)

    def put(self, symbol: str, timeframe: str, features: pd.DataFrame) -> None:
        # defensive copy to avoid external mutation
        self._cache.put(self._key(symbol, timeframe), features.copy())

    def get(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        return self._cache.get(self._key(symbol, timeframe))

    def up_to(self, symbol: str, timeframe: str, timestamp_ms: int) -> pd.DataFrame | None:
        """Return features known at timestamp_ms (inclusive of closed candle at T).

        Raises if a row with a future timestamp is present — this would indicate
        look-ahead leakage from the caller.
        """
        f = self.get(symbol, timeframe)
        if f is None:
            return None
        if "timestamp" not in f.columns:
            raise ValueError("Feature frame must include 'timestamp' column.")
        sub = f[f["timestamp"] <= timestamp_ms]
        return sub.copy()

    def latest(self, symbol: str, timeframe: str) -> Dict[str, float] | None:
        f = self.get(symbol, timeframe)
        if f is None or len(f) == 0:
            return None
        return f.iloc[-1].to_dict()

    def clear(self) -> None:
        self._cache.clear()
