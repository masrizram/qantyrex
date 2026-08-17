"""Momentum engine: RSI/MACD/ADX state with configurable thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from .indicators import rsi as calc_rsi, macd as calc_macd, adx as calc_adx


@dataclass
class MomentumConfig:
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    adx_period: int = 14
    adx_min: float = 20.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9


def add_momentum(df: pd.DataFrame, cfg: MomentumConfig | None = None) -> pd.DataFrame:
    cfg = cfg or MomentumConfig()
    out = df.copy()
    out["rsi"] = calc_rsi(out["close"], cfg.rsi_period)
    macd_line, signal_line, hist = calc_macd(
        out["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    out["adx"] = calc_adx(out, cfg.adx_period)
    # RSI rising/falling relative to previous bar (no look-ahead)
    out["rsi_prev"] = out["rsi"].shift(1)
    return out


def buy_momentum(row: pd.Series, cfg: MomentumConfig) -> Tuple[bool, str]:
    """Return (passes, reason). All thresholds configurable."""
    if pd.isna(row.get("rsi")) or pd.isna(row.get("adx")) or pd.isna(row.get("macd")):
        return False, "momentum_warmup"
    if row["rsi"] <= cfg.rsi_oversold:
        return False, "rsi_oversold"  # not yet recovering
    if row["rsi"] <= row.get("rsi_prev", row["rsi"]):
        return False, "rsi_not_rising"
    if row["macd"] <= row["macd_signal"]:
        return False, "macd_not_bullish"
    if row["adx"] < cfg.adx_min:
        return False, "adx_below_min"
    return True, "ok"


def sell_momentum(row: pd.Series, cfg: MomentumConfig) -> Tuple[bool, str]:
    if pd.isna(row.get("rsi")) or pd.isna(row.get("adx")) or pd.isna(row.get("macd")):
        return False, "momentum_warmup"
    if row["rsi"] >= cfg.rsi_overbought:
        return False, "rsi_overbought"
    if row["rsi"] >= row.get("rsi_prev", row["rsi"]):
        return False, "rsi_not_falling"
    if row["macd"] >= row["macd_signal"]:
        return False, "macd_not_bearish"
    if row["adx"] < cfg.adx_min:
        return False, "adx_below_min"
    return True, "ok"
