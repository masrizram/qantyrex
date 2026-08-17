"""Trend classification engine."""
from __future__ import annotations

import pandas as pd

from ..core.enums import TrendState
from .indicators import ema, ema_slope


def classify_trend(
    df: pd.DataFrame,
    ema_fast: int = 50,
    ema_slow: int = 200,
    slope_lookback: int = 5,
) -> pd.DataFrame:
    """Append EMA columns and a `trend` classification to df.

    Classification logic (for each bar, using only closed data up to that bar):
      - price > EMA200, EMA50 > EMA200, EMA50 slope up        -> STRONG_BULLISH
      - price > EMA200, EMA50 > EMA200 (slope flat/down)      -> BULLISH
      - price < EMA200, EMA50 < EMA200, EMA50 slope down      -> STRONG_BEARISH
      - price < EMA200, EMA50 < EMA200 (slope flat/up)        -> BEARISH
      - otherwise                                             -> NEUTRAL
    """
    out = df.copy()
    out["ema_fast"] = ema(out["close"], ema_fast)
    out["ema_slow"] = ema(out["close"], ema_slow)
    out["ema_fast_slope"] = ema_slope(out["ema_fast"], slope_lookback)
    price = out["close"]
    ef = out["ema_fast"]
    es = out["ema_slow"]
    sl = out["ema_fast_slope"]

    def _cls(row):
        if pd.isna(row["ema_fast"]) or pd.isna(row["ema_slow"]):
            return TrendState.NEUTRAL
        p = row["close"]
        if p > row["ema_slow"] and row["ema_fast"] > row["ema_slow"]:
            if row["ema_fast_slope"] > 0:
                return TrendState.STRONG_BULLISH
            return TrendState.BULLISH
        if p < row["ema_slow"] and row["ema_fast"] < row["ema_slow"]:
            if row["ema_fast_slope"] < 0:
                return TrendState.STRONG_BEARISH
            return TrendState.BEARISH
        return TrendState.NEUTRAL

    out["trend"] = out.apply(_cls, axis=1).map(lambda x: x.value if isinstance(x, TrendState) else x)
    out["trend_state"] = out["trend"].map(
        lambda v: TrendState(v) if isinstance(v, str) else v
    )
    return out


def is_bullish(trend: TrendState) -> bool:
    return trend in (TrendState.STRONG_BULLISH, TrendState.BULLISH)


def is_bearish(trend: TrendState) -> bool:
    return trend in (TrendState.STRONG_BEARISH, TrendState.BEARISH)
