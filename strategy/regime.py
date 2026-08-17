"""Market regime classifier, independently testable."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.enums import RegimeState, RegimeAction
from .indicators import adx, atr_percent, rolling_volatility


REGIME_ACTIONS = {
    RegimeState.STRONG_TREND: RegimeAction.TRADE,
    RegimeState.WEAK_TREND: RegimeAction.CONDITIONAL,
    RegimeState.RANGE: RegimeAction.NO_TRADE,
    RegimeState.HIGH_VOLATILITY: RegimeAction.NO_TRADE,
    RegimeState.LOW_VOLATILITY: RegimeAction.NO_TRADE,
    RegimeState.TRANSITION: RegimeAction.NO_TRADE,
    RegimeState.UNKNOWN: RegimeAction.NO_TRADE,
}


@dataclass
class RegimeConfig:
    adx_period: int = 14
    adx_strong: float = 25.0
    adx_weak: float = 20.0
    atr_period: int = 14
    atr_high_percentile: float = 85.0
    atr_low_percentile: float = 15.0
    rolling_window: int = 20
    rank_window: int = 100


def classify_regime(df: pd.DataFrame, cfg: RegimeConfig | None = None) -> pd.DataFrame:
    cfg = cfg or RegimeConfig()
    out = df.copy()
    out["adx"] = adx(out, cfg.adx_period)
    out["atr_percent"] = atr_percent(out, cfg.atr_period)
    out["rolling_vol"] = rolling_volatility(out["close"], cfg.rolling_window)
    out["atr_pct_rank"] = out["atr_percent"].rolling(
        cfg.rank_window, min_periods=cfg.rank_window).rank(pct=True) * 100.0

    def _row(row):
        if pd.isna(row["adx"]) or pd.isna(row["atr_pct_rank"]):
            return RegimeState.UNKNOWN
        # Volatility takes priority as a disqualifier
        if row["atr_pct_rank"] >= cfg.atr_high_percentile:
            return RegimeState.HIGH_VOLATILITY
        if row["atr_pct_rank"] <= cfg.atr_low_percentile:
            return RegimeState.LOW_VOLATILITY
        # Trend strength
        if row["adx"] >= cfg.adx_strong:
            return RegimeState.STRONG_TREND
        if row["adx"] >= cfg.adx_weak:
            return RegimeState.WEAK_TREND
        # If ADX is rising fast (transition) -> TRANSITION, else RANGE
        return RegimeState.RANGE

    states = []
    for _, row in out.iterrows():
        states.append(_row(row))
    out["regime"] = [s.value for s in states]
    out["regime_state"] = [s for s in states]
    out["regime_action"] = [REGIME_ACTIONS[s].value for s in states]
    return out


def action_for(regime: RegimeState) -> RegimeAction:
    return REGIME_ACTIONS[regime]
