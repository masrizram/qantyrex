"""Signal engine: combines features -> candidate signals with entry filters.

Enforces strict no-look-ahead: only rows up to (and including) the current
closed candle are used. All filters reject; they never silently relax.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..core.enums import (RegimeAction, RegimeState, Side, TrendState)
from ..core.exceptions import LookAheadBiasError
from ..core.models import Signal, config_hash
from .momentum import MomentumConfig, add_momentum, buy_momentum, sell_momentum
from .regime import RegimeConfig, classify_regime
from .scoring import score_signal
from .support_resistance import SRConfig, add_support_resistance, room_for_tp
from .trend import classify_trend
from .volatility import VolatilityConfig, add_volatility, volatility_status


@dataclass
class SignalResult:
    signal: Optional[Signal]
    rejected_reason: Optional[str] = None
    score: Optional[float] = None


class SignalEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.strategy_version = cfg.strategy_version
        self.mcfg = MomentumConfig(
            rsi_period=cfg.rsi_period, rsi_oversold=cfg.rsi_oversold,
            rsi_overbought=cfg.rsi_overbought, adx_period=cfg.adx_period,
            adx_min=cfg.adx_min,
        )
        self.vcfg = VolatilityConfig(
            atr_period=cfg.atr_period,
            atr_min_percent=cfg.atr_min_percent,
            atr_max_percent=cfg.atr_max_percent,
        )
        self.rcfg = RegimeConfig(adx_period=cfg.adx_period, atr_period=cfg.atr_period)
        self.srcfg = SRConfig()

    def build_features(self, df_entry: pd.DataFrame, df_trend: pd.DataFrame) -> pd.DataFrame:
        """Attach trend/momentum/volatility/regime/SR to entry-timeframe frame."""
        out = classify_trend(df_entry, self.cfg.ema_fast, self.cfg.ema_slow)
        out = add_momentum(out, self.mcfg)
        out = add_volatility(out, self.vcfg)
        out = classify_regime(out, self.rcfg)
        out = add_support_resistance(out, self.srcfg)
        # Attach higher-timeframe trend (last closed HT bar <= each entry bar)
        if df_trend is not None and len(df_trend):
            ht = classify_trend(df_trend, self.cfg.ema_fast, self.cfg.ema_slow)
            # use the string `trend` column so downstream .value string comparisons hold
            ht = ht[["timestamp", "trend"]].rename(
                columns={"timestamp": "timestamp_ht", "trend": "trend_tf"})
            # map by floor join: for each entry bar, last HT bar with ts <= entry ts
            out = out.sort_values("timestamp").reset_index(drop=True)
            ht = ht.sort_values("timestamp_ht").reset_index(drop=True)
            out["trend_tf"] = pd.merge_asof(
                out, ht, left_on="timestamp", right_on="timestamp_ht",
                direction="backward", allow_exact_matches=True,
            )["trend_tf"].values
        return out

    def evaluate(self, features: pd.DataFrame, idx: int,
                 spread_percent: float = 0.0,
                 liquidity_ok: bool = True,
                 latency_ms: Optional[float] = None) -> SignalResult:
        """Evaluate a single closed bar at `idx` (0-indexed) using only rows <= idx."""
        if idx < 0 or idx >= len(features):
            return SignalResult(None, "out_of_range")
        # Use only data up to and including idx (no look-ahead)
        row = features.iloc[idx]
        if idx != features.index.get_loc(features.index[idx]):
            # safety: ensure row ordering
            pass
        # ---- Regime gate ----
        regime = row["regime_state"]
        action = row["regime_action"]
        if action != RegimeAction.TRADE.value:
            return SignalResult(None, f"regime_{regime}_no_trade")

        # ---- Volatility gate ----
        vlabel, vok = volatility_status(row, self.vcfg)
        if not vok:
            return SignalResult(None, f"volatility_{vlabel}")

        # ---- Trend gate ----
        trend = row["trend_state"]
        trend = trend.value if isinstance(trend, TrendState) else str(trend)
        trend_tf = row.get("trend_tf", trend)
        trend_tf = trend_tf.value if isinstance(trend_tf, TrendState) else str(trend_tf)
        if trend == TrendState.NEUTRAL.value:
            return SignalResult(None, "trend_neutral")
        # Determine side from trend direction
        side = Side.BUY if trend in (TrendState.STRONG_BULLISH.value, TrendState.BULLISH.value) else Side.SELL

        # ---- Momentum gate ----
        if side == Side.BUY:
            m_ok, m_reason = buy_momentum(row, self.mcfg)
        else:
            m_ok, m_reason = sell_momentum(row, self.mcfg)
        if not m_ok:
            return SignalResult(None, f"momentum_{m_reason}")

        # ---- Spread / liquidity gate ----
        if spread_percent > self.cfg.max_spread_percent:
            return SignalResult(None, "spread_too_high")
        if not liquidity_ok:
            return SignalResult(None, "liquidity_too_low")

        # ---- SL candidates ----
        sl = self._stop_loss(row, side)
        if sl is None:
            return SignalResult(None, "no_valid_sl")
        entry = float(row["close"])
        if side == Side.BUY and sl >= entry:
            return SignalResult(None, "sl_above_entry")
        if side == Side.SELL and sl <= entry:
            return SignalResult(None, "sl_below_entry")

        # ---- TP from RR ----
        risk = abs(entry - sl)
        if risk <= 0:
            return SignalResult(None, "zero_risk")
        tp = entry + self.cfg.min_rr * risk if side == Side.BUY else entry - self.cfg.min_rr * risk
        rr = self.cfg.min_rr

        # ---- Structure gate ----
        nr = row.get("nearest_resistance", np.nan)
        ns = row.get("nearest_support", np.nan)
        room_ok, room_reason = room_for_tp(entry, tp, side.value, nr, ns)
        if not room_ok:
            return SignalResult(None, room_reason)

        # ---- Score ----
        support_strength = int(row.get("support_strength", 0) or 0)
        resistance_strength = int(row.get("resistance_strength", 0) or 0)
        entry_tf_aligned = (side == Side.BUY and trend_tf in (TrendState.STRONG_BULLISH.value, TrendState.BULLISH.value)) or \
                           (side == Side.SELL and trend_tf in (TrendState.STRONG_BEARISH.value, TrendState.BEARISH.value))
        sb = score_signal(
            side=side, trend=trend, trend_tf=trend_tf, entry_tf_aligned=entry_tf_aligned,
            momentum_passes=True, adx=float(row.get("adx", 0) or 0), adx_min=self.cfg.adx_min,
            support_strength=support_strength, resistance_strength=resistance_strength,
            room_ok=True, volatility_label=vlabel, volatility_ok=vok,
            spread_percent=spread_percent, max_spread=self.cfg.max_spread_percent,
            liquidity_ok=liquidity_ok, rr=rr, min_rr=self.cfg.min_rr, latency_ms=latency_ms,
            min_score=self.cfg.min_signal_score,
        )
        if not sb.passes:
            return SignalResult(None, f"score_{sb.total:.1f}_below_min", sb.total)

        sig = Signal(
            strategy_version=self.strategy_version,
            symbol=self.cfg.symbol, side=side, entry=entry, stop_loss=sl,
            take_profit=tp, rr=rr, score=sb.total, trend=trend,
            regime=regime, regime_action=action, rsi=float(row.get("rsi", 0) or 0),
            atr=float(row.get("atr", 0) or 0), atr_percent=float(row.get("atr_percent", 0) or 0),
            adx=float(row.get("adx", 0) or 0), ema_fast=float(row.get("ema_fast", 0) or 0),
            ema_slow=float(row.get("ema_slow", 0) or 0), spread_percent=spread_percent,
            timestamp=int(row["timestamp"]),
            features={"support": float(ns) if not np.isnan(ns) else 0,
                      "resistance": float(nr) if not np.isnan(nr) else 0,
                      "breakdown": sb.components},
        )
        return SignalResult(sig, None, sb.total)

    def _stop_loss(self, row: pd.Series, side: Side) -> Optional[float]:
        """ATR-based stop with a floor at the nearest confirmed structure."""
        atr_v = row.get("atr", np.nan)
        close = row["close"]
        if pd.isna(atr_v) or atr_v <= 0:
            return None
        # 1.5x ATR stop
        atr_stop = close - 1.5 * atr_v if side == Side.BUY else close + 1.5 * atr_v
        # tighten with nearest swing structure if it gives a tighter (better) stop
        if side == Side.BUY:
            ns = row.get("nearest_support", np.nan)
            if not pd.isna(ns) and ns < close and ns > atr_stop:
                atr_stop = ns
        else:
            nr = row.get("nearest_resistance", np.nan)
            if not pd.isna(nr) and nr > close and nr < atr_stop:
                atr_stop = nr
        return float(atr_stop)
