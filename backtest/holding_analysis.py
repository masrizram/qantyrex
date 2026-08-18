"""Holding-period analysis (Phase 25-27).

Instruments the simulator to record per-bar position state, distance to SL/TP,
ATR, and regime, so we can determine exactly why a position stays open for
extended periods.

DIAGNOSTIC ONLY. Does NOT modify the strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..core.enums import Side, ExitReason, TrendState, RegimeState
from ..core.models import Position, Signal
from ..strategy.indicators import atr, ema
from ..strategy.trend import classify_trend
from ..strategy.regime import classify_regime, RegimeConfig
from ..strategy.momentum import MomentumConfig, add_momentum
from ..backtest.simulator import Simulator
from ..backtest.simulator import SimulatorConfig


@dataclass
class BarState:
    bar_idx: int
    timestamp: int
    close: float
    atr: float
    sl: float
    tp: float
    dist_to_sl_pct: float
    dist_to_tp_pct: float
    sl_updated: bool
    regime: str
    trend: str
    max_favorable: float
    max_adverse: float


@dataclass
class HoldingAnalysis:
    position_id: str
    entry_bar: int
    exit_bar: int
    holding_bars: int
    side: str
    entry_price: float
    exit_price: float
    exit_reason: str
    bar_states: List[BarState] = field(default_factory=list)
    sl_updates: int = 0
    sl_initial: float = 0.0
    sl_final: float = 0.0
    tp_initial: float = 0.0
    pnl: float = 0.0

    def summary(self) -> str:
        lines = [
            f"HOLDING ANALYSIS  id={self.position_id}  side={self.side}",
            f"  holding_bars={self.holding_bars}  entry={self.entry_price:.2f}  exit={self.exit_price:.2f}  pnl={self.pnl:.2f}",
            f"  exit_reason={self.exit_reason}  sl_initial={self.sl_initial:.2f}  sl_final={self.sl_final:.2f}  sl_updates={self.sl_updates}",
            f"  tp_initial={self.tp_initial:.2f}",
        ]
        if self.bar_states:
            distances = [b.dist_to_tp_pct for b in self.bar_states]
            lines.append(f"  dist_to_tp: min={min(distances):.2f}% max={max(distances):.2f}% mean={np.mean(distances):.2f}%")
            regimes = set(b.regime for b in self.bar_states)
            lines.append(f"  regimes_observed: {regimes}")
            # how close did price get to TP?
            closest = min(distances)
            lines.append(f"  closest_to_tp: {closest:.2f}%")
        return "\n".join(lines)


def analyze_holding_period(
    cfg: Config,
    df: pd.DataFrame,
    accepted_signals: List[Signal],
    features: pd.DataFrame,
) -> List[HoldingAnalysis]:
    from ..backtest.simulator import Simulator, SimulatorConfig

    # Run the simulator TRADE-BY-TRADE with max_open=99 so all signals get a chance
    # to open (not just the first). This reveals the true holding period for each.
    sim = Simulator(SimulatorConfig(
        fee_rate=cfg.fee_rate, slippage_bps=cfg.slippage_bps,
        break_even_r=cfg.break_even_r, max_open_positions=99,
        seed=0, initial_equity=999_000.0,  # high equity to avoid DD gate
    ))
    sim_result = sim.run(df, accepted_signals, features_for_atr=features)
    trades = sim_result.trades

    analyses: List[HoldingAnalysis] = []
    ts_to_idx = {int(t): i for i, t in enumerate(df["timestamp"])}

    for _, tr in trades.iterrows():
        opened_at = tr.get("opened_at")
        closed_at = tr.get("closed_at")
        if opened_at is None or closed_at is None:
            # Fallback: use legacy timestamp (exit ts) as closed_at, and
            # try to find opened_at from the signal timestamp if available.
            continue

        entry_idx = ts_to_idx.get(int(opened_at))
        exit_idx = ts_to_idx.get(int(closed_at))
        if entry_idx is None or exit_idx is None:
            continue

        if closed_at < opened_at:
            continue

        side = Side.BUY if tr["side"] == "BUY" else Side.SELL
        entry_price = float(tr.get("entry_price", tr.get("entry", 0)))
        initial_sl = float(tr.get("initial_stop_loss", tr.get("stop_loss", 0)))
        final_sl = float(tr.get("final_stop_loss", tr.get("stop_loss", 0)))
        initial_tp = float(tr["take_profit"])

        holding_bars = exit_idx - entry_idx

        bar_states = _replay_position(
            df, features, cfg, entry_idx, exit_idx, entry_price,
            initial_sl, initial_tp, side,
        )

        sl_updates = sum(1 for b in bar_states if b.sl_updated)
        sl_final_state = bar_states[-1].sl if bar_states else initial_sl

        analyses.append(HoldingAnalysis(
            position_id=tr.get("trade_id", ""),
            entry_bar=entry_idx, exit_bar=exit_idx,
            holding_bars=holding_bars, side=tr["side"],
            entry_price=entry_price, exit_price=float(tr.get("exit_price", tr.get("exit", 0))),
            exit_reason=tr.get("exit_reason", ""),
            bar_states=bar_states, sl_updates=sl_updates,
            sl_initial=initial_sl, sl_final=final_sl,
            tp_initial=initial_tp, pnl=float(tr.get("pnl", 0)),
        ))
    return analyses


def _replay_position(
    df: pd.DataFrame, features: pd.DataFrame, cfg: Config,
    start_idx: int, end_idx: int, entry: float, sl: float, tp: float,
    side: Side,
) -> List[BarState]:
    """Replay a position bar-by-bar, tracking SL/TP/BE/trailing, to produce
    per-bar state for holding-period analysis."""
    bar_states: List[BarState] = []
    moved_be = False
    trail_mult = 2.0
    break_even_r = cfg.break_even_r
    risk = abs(entry - sl)

    for i in range(start_idx + 1, min(end_idx + 1, len(df))):
        bar = df.iloc[i]
        hi, lo, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        atr_v = features["atr"].iloc[i] if i < len(features) and "atr" in features.columns else 0.0

        regime = "unknown"
        if "regime" in features.columns:
            regime = features["regime"].iloc[i] if i < len(features) else "unknown"

        trend = "unknown"
        if "trend_state" in features.columns:
            ts = features["trend_state"].iloc[i]
            trend = ts.value if hasattr(ts, "value") else str(ts)

        sl_before = sl
        if side == Side.BUY:
            fav = (hi - entry)
            adv = (lo - entry)
            if not moved_be and fav >= risk * break_even_r:
                sl = max(sl, entry)
                moved_be = True
            if atr_v and not np.isnan(atr_v):
                new_trail = hi - trail_mult * atr_v
                if new_trail > sl:
                    sl = new_trail
        else:
            fav = (entry - lo)
            adv = (entry - hi)
            if not moved_be and fav >= risk * break_even_r:
                sl = min(sl, entry)
                moved_be = True
            if atr_v and not np.isnan(atr_v):
                new_trail = lo + trail_mult * atr_v
                if new_trail < sl:
                    sl = new_trail

        dist_sl = abs(close - sl) / close * 100
        dist_tp = abs(tp - close) / close * 100

        bar_states.append(BarState(
            bar_idx=i, timestamp=int(bar["timestamp"]), close=close,
            atr=float(atr_v) if not np.isnan(atr_v) else 0.0,
            sl=sl, tp=tp, dist_to_sl_pct=dist_sl, dist_to_tp_pct=dist_tp,
            sl_updated=sl != sl_before,
            regime=regime, trend=trend,
            max_favorable=fav, max_adverse=adv,
        ))
    return bar_states


def holding_period_bottleneck_report(analyses: List[HoldingAnalysis]) -> str:
    """Produce a human-readable bottleneck report."""
    if not analyses:
        return "NO POSITIONS ANALYZED"
    lines = []
    for a in analyses:
        lines.append(a.summary())
        if a.bar_states:
            bs = a.bar_states
            dist_to_tp = [b.dist_to_tp_pct for b in bs]
            lines.append(f"  TP_DISTANCE: min={min(dist_to_tp):.2f}% max={max(dist_to_tp):.2f}% mean={np.mean(dist_to_tp):.2f}%")
            lines.append(f"  SL_DISTANCE: min={min(b.dist_to_sl_pct for b in bs):.2f}% max={max(b.dist_to_sl_pct for b in bs):.2f}%")
            # How many bars were within 1% of TP?
            near_tp = sum(1 for d in dist_to_tp if d < 1.0)
            lines.append(f"  BARS_WITHIN_1%_OF_TP: {near_tp}/{len(bs)} ({near_tp/len(bs)*100:.1f}%)")
            # How many bars had SL updates?
            lines.append(f"  SL_UPDATES: {a.sl_updates} (BE + trailing)")
            # Did the SL ever reach the entry (break-even)?
            if a.side == "BUY":
                lines.append(f"  BE_ACTIVATED: {a.sl_final >= a.entry_price}")
            else:
                lines.append(f"  BE_ACTIVATED: {a.sl_final <= a.entry_price}")
            # Regime distribution
            from collections import Counter
            reg_counts = Counter(b.regime for b in bs)
            lines.append(f"  REGIMES: {dict(reg_counts.most_common(5))}")
    return "\n".join(lines)