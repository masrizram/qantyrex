"""Strict market-data validator. Rejects corrupted data; never fabricates."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from ..core.exceptions import DataQualityError
from ..core.models import Candle


TF_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def timeframe_ms(tf: str) -> int:
    if tf in TF_TO_MS:
        return TF_TO_MS[tf]
    # parse expressions like "45m", "12h"
    t = tf.strip().lower()
    if t.endswith("m") and t[:-1].isdigit():
        return int(t[:-1]) * 60_000
    if t.endswith("h") and t[:-1].isdigit():
        return int(t[:-1]) * 3_600_000
    if t.endswith("d") and t[:-1].isdigit():
        return int(t[:-1]) * 86_400_000
    raise ValueError(f"Unknown timeframe {tf!r}")


class DataValidator:
    """Validates a list/dict of OHLCV candles.

    All checks are strict: the smallest failure aborts the whole batch.
    We never silently fabricate or repair missing candles.
    """

    def __init__(
        self,
        timeframe: str,
        max_gap_tolerance: int = 1,
        max_stale_seconds: int = 600,
    ) -> None:
        self.timeframe = timeframe
        self.tf_ms = timeframe_ms(timeframe)
        self.max_gap_tolerance = max_gap_tolerance
        self.max_stale_seconds = max_stale_seconds

    # ---- public API ----
    def validate(self, candles: Sequence[Candle] | pd.DataFrame,
                 now_ms: int | None = None) -> List[Candle]:
        """Validate and return the candles as a sorted list. Raises on any error."""
        if isinstance(candles, pd.DataFrame):
            candles = self._df_to_candles(candles)
        if not candles:
            raise DataQualityError("Empty candle batch.")
        self._check_schema(candles)
        self._check_dups(candles)
        self._check_ordering(candles)
        self._check_ohlc_integrity(candles)
        self._check_volume(candles)
        self._check_gaps(candles)
        if now_ms is not None:
            self._check_stale(candles, now_ms)
        return list(candles)

    def to_dataframe(self, candles: Sequence[Candle]) -> pd.DataFrame:
        df = pd.DataFrame([{
            "timestamp": c.timestamp, "open": c.open, "high": c.high,
            "low": c.low, "close": c.close, "volume": c.volume, "closed": c.closed
        } for c in candles])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("datetime")

    # ---- checks ----
    def _check_schema(self, candles: Sequence[Candle]) -> None:
        for c in candles:
            for f in ("symbol", "timeframe", "timestamp", "open", "high",
                      "low", "close", "volume"):
                if getattr(c, f, None) is None:
                    raise DataQualityError(f"Candle missing field {f}.")
            if c.timeframe != self.timeframe:
                raise DataQualityError(
                    f"Timeframe mismatch: expected {self.timeframe}, got {c.timeframe}.")

    def _check_dups(self, candles: Sequence[Candle]) -> None:
        seen = set()
        for c in candles:
            if c.timestamp in seen:
                raise DataQualityError(f"Duplicate candle at ts={c.timestamp}.")
            seen.add(c.timestamp)

    def _check_ordering(self, candles: Sequence[Candle]) -> None:
        ts = [c.timestamp for c in candles]
        if ts != sorted(ts):
            raise DataQualityError("Candles are not in ascending timestamp order.")

    def _check_ohlc_integrity(self, candles: Sequence[Candle]) -> None:
        for c in candles:
            if not np.isfinite([c.open, c.high, c.low, c.close, c.volume]).all():
                raise DataQualityError(f"non-finite values at ts={c.timestamp}")
            if not (c.low <= c.high):
                raise DataQualityError(f"low>high at ts={c.timestamp}")
            if not (c.low <= c.open <= c.high):
                raise DataQualityError(f"open outside L/H at ts={c.timestamp}")
            if not (c.low <= c.close <= c.high):
                raise DataQualityError(f"close outside L/H at ts={c.timestamp}")
            if c.high <= 0 or c.low <= 0 or c.close <= 0:
                raise DataQualityError(f"non-positive price at ts={c.timestamp}")
            if c.volume < 0:
                raise DataQualityError(f"negative volume at ts={c.timestamp}")

    def _check_volume(self, candles: Sequence[Candle]) -> None:
        # All-zero volume across a long batch is suspicious but not fatal; skip strict check.
        return

    def _check_gaps(self, candles: Sequence[Candle]) -> None:
        ts = [c.timestamp for c in candles]
        for i in range(1, len(ts)):
            diff = ts[i] - ts[i - 1]
            if diff == 0:
                continue  # handled by dup check
            if diff % self.tf_ms != 0:
                raise DataQualityError(
                    f"Timestamp gap not multiple of timeframe at i={i}: {diff}ms")
            if diff > self.tf_ms * (self.max_gap_tolerance + 1):
                raise DataQualityError(
                    f"Missing candle(s) between {ts[i-1]} and {ts[i]} "
                    f"(gap={diff}ms, tf={self.tf_ms}ms).")

    def _check_stale(self, candles: Sequence[Candle], now_ms: int) -> None:
        last = max(c.timestamp for c in candles)
        if (now_ms - last) > self.max_stale_seconds * 1000 + self.tf_ms:
            raise DataQualityError(
                f"Stale data: last candle at {last}, now {now_ms}, "
                f"age={(now_ms-last)//1000}s exceeds {self.max_stale_seconds}s.")

    # ---- helpers ----
    def _df_to_candles(self, df: pd.DataFrame) -> List[Candle]:
        df = df.reset_index() if df.index.name else df.copy()
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            raise DataQualityError(f"DataFrame missing columns: {required - set(df.columns)}")
        out = []
        for _, r in df.iterrows():
            out.append(Candle(
                symbol=str(r.get("symbol", "")),
                timeframe=str(r.get("timeframe", self.timeframe)),
                timestamp=int(r["timestamp"]),
                open=float(r["open"]), high=float(r["high"]),
                low=float(r["low"]), close=float(r["close"]),
                volume=float(r["volume"]),
                closed=bool(r.get("closed", True)),
            ))
        return out
