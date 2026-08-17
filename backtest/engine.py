"""Backtest engine: drives the signal engine + risk manager + simulator
over a candle series, producing a single end-to-end simulation.

Supports data splits (TRAIN/VALIDATION/OOS) and no-look-ahead enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from ..config import Config
from ..core.enums import Side
from ..core.models import Signal
from ..risk.risk_manager import RiskManager
from ..strategy.signal_engine import SignalEngine
from .simulator import Simulator, SimulatorConfig, SimulationResult


@dataclass
class BacktestResult:
    result: SimulationResult
    split: str
    n_candles: int
    n_signals: int


def split_data(df: pd.DataFrame, fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2)
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split. TRAIN / VALIDATION / OOS."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1.0")
    n = len(df)
    n_tr = int(n * fractions[0])
    n_va = int(n * fractions[1])
    train = df.iloc[:n_tr].copy()
    val = df.iloc[n_tr:n_tr + n_va].copy()
    oos = df.iloc[n_tr + n_va:].copy()
    return train, val, oos


class Backtester:
    def __init__(self, cfg: Config, sim_cfg: SimulatorConfig | None = None) -> None:
        self.cfg = cfg
        self.sim_cfg = sim_cfg or SimulatorConfig(
            fee_rate=cfg.fee_rate, slippage_bps=cfg.slippage_bps,
            break_even_r=cfg.break_even_r,
            max_open_positions=cfg.max_open_positions,
        )
        self.engine = SignalEngine(cfg)
        self.sim = Simulator(self.sim_cfg)

    def run(self, df: pd.DataFrame, split: str = "FULL",
            spread_percent: float = 0.0,
            liquidity_ok: bool = True,
            risk_manager: Optional[RiskManager] = None,
            start_equity: float = 10_000.0) -> BacktestResult:
        """Generate signals bar-by-bar (no look-ahead) and simulate them."""
        features = self.engine.build_features(df, df)
        signals: List[Signal] = []
        rm = risk_manager or RiskManager(self.cfg, equity=start_equity)
        for i in range(len(features)):
            res = self.engine.evaluate(features, idx=i, spread_percent=spread_percent,
                                       liquidity_ok=liquidity_ok)
            if res.signal is None:
                continue
            # risk check + sizing
            decision = rm.evaluate_signal(res.signal, open_positions=0)
            if not decision.allowed or decision.sizing is None:
                continue
            sig = res.signal
            # embed the sized quantity into the signal for the simulator
            sig.features = dict(sig.features or {})
            sig.features["quantity"] = decision.sizing.quantity
            signals.append(sig)
        # reset simulator equity to the risk manager's current equity
        sim = Simulator(SimulatorConfig(
            fee_rate=self.sim_cfg.fee_rate, slippage_bps=self.sim_cfg.slippage_bps,
            spread_bps=self.sim_cfg.spread_bps, latency_bars=self.sim_cfg.latency_bars,
            rejection_prob=self.sim_cfg.rejection_prob,
            partial_fill_prob=self.sim_cfg.partial_fill_prob,
            break_even_r=self.sim_cfg.break_even_r,
            trail_atr_mult=self.sim_cfg.trail_atr_mult,
            max_open_positions=self.sim_cfg.max_open_positions,
            seed=self.sim_cfg.seed,
            initial_equity=start_equity,
        ))
        result = sim.run(df, signals, features_for_atr=features)
        return BacktestResult(result=result, split=split,
                              n_candles=len(df), n_signals=len(signals))
