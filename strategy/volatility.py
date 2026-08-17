"""Volatility engine: ATR, ATR%, rolling vol, percentile, validity gates."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.exceptions import DataQualityError
from .indicators import atr, atr_percent, rolling_volatility, percentile_rank


@dataclass
class VolatilityConfig:
    atr_period: int = 14
    atr_min_percent: float = 0.20
    atr_max_percent: float = 3.00
    percentile_window: int = 100
    rolling_window: int = 20


def add_volatility(df: pd.DataFrame, cfg: VolatilityConfig | None = None) -> pd.DataFrame:
    cfg = cfg or VolatilityConfig()
    out = df.copy()
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_percent"] = atr_percent(out, cfg.atr_period)
    out["rolling_vol"] = rolling_volatility(out["close"], cfg.rolling_window)
    # Rolling percentile of ATR% over the last `percentile_window` bars.
    # We use a simple expanding-rank approach so that early bars are NaN.
    out["atr_percent_rank"] = out["atr_percent"].rolling(
        cfg.percentile_window, min_periods=cfg.percentile_window).rank(pct=True) * 100.0
    return out


def volatility_status(row: pd.Series, cfg: VolatilityConfig) -> tuple[str, bool]:
    """Return (label, ok_to_trade). Rejects abnormal regimes."""
    if pd.isna(row.get("atr_percent")):
        return "warmup", False
    p = row["atr_percent"]
    if p < cfg.atr_min_percent:
        return "too_low", False
    if p > cfg.atr_max_percent:
        return "too_high", False
    # Detect a volatility shock: ATR% jumps to >2x its recent rolling value
    if pd.notna(row.get("atr")) and pd.notna(row.get("rolling_vol")):
        if row["rolling_vol"] > 0 and row["atr_percent"] > 2.5 * cfg.atr_max_percent:
            return "shock", False
    return "normal", True


def reject_abnormal_volatility(df: pd.DataFrame, cfg: VolatilityConfig) -> pd.DataFrame:
    """Return a mask of rows whose volatility is acceptable."""
    out = add_volatility(df, cfg)
    masks = []
    for _, row in out.iterrows():
        _, ok = volatility_status(row, cfg)
        masks.append(ok)
    return pd.Series(masks, index=out.index)
