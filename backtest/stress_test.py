"""Stress testing: simulated adversarial conditions against the system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from .simulator import SimulatorConfig, Simulator, SimulationResult


@dataclass
class StressScenario:
    name: str
    sim_cfg: SimulatorConfig
    spread_bps_override: float | None = None
    slippage_bps_override: float | None = None
    fee_rate_override: float | None = None
    candles_mod: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    description: str = ""


def _volatility_spike(df: pd.DataFrame, idx: int, factor: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    cols = ["high", "low"]
    for c in cols:
        df.iloc[idx, df.columns.get_loc(c)] = df[c].iloc[idx] * (factor if c == "high" else 1.0 / factor)
    return df


def default_scenarios() -> List[StressScenario]:
    return [
        StressScenario("2x_spread", SimulatorConfig(spread_bps=2.0), spread_bps_override=2.0,
                       description="Double the spread"),
        StressScenario("3x_spread", SimulatorConfig(spread_bps=3.0), spread_bps_override=3.0,
                       description="Triple the spread"),
        StressScenario("2x_slippage", SimulatorConfig(slippage_bps=4.0), slippage_bps_override=4.0,
                       description="Double the slippage"),
        StressScenario("3x_slippage", SimulatorConfig(slippage_bps=6.0), slippage_bps_override=6.0,
                       description="Triple the slippage"),
        StressScenario("higher_fees", SimulatorConfig(fee_rate=0.003), fee_rate_override=0.003,
                       description="3x maker/taker fees"),
        StressScenario("latency_2bars", SimulatorConfig(latency_bars=2),
                       description="2-bar execution latency"),
        StressScenario("order_rejections", SimulatorConfig(rejection_prob=0.2),
                       description="20% of orders rejected"),
        StressScenario("partial_fills", SimulatorConfig(partial_fill_prob=0.3,
                                                        partial_fill_ratio=0.5),
                       description="30% partial fills at 50% ratio"),
    ]


def run_stress(candles: pd.DataFrame, signals, features,
               scenarios: List[StressScenario] | None = None,
               initial_equity: float = 10_000.0) -> Dict[str, SimulationResult]:
    """Run each stress scenario through the simulator with the same signals."""
    scenarios = scenarios or default_scenarios()
    out: Dict[str, SimulationResult] = {}
    for sc in scenarios:
        sim = Simulator(sc.sim_cfg)
        if sc.candles_mod is not None:
            df = sc.candles_mod(candles)
        else:
            df = candles
        out[sc.name] = sim.run(df, signals, features_for_atr=features)
    return out


def stress_summary(results: Dict[str, SimulationResult]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for name, r in results.items():
        m = r.metrics
        summary[name] = {
            "net_profit": m.net_profit, "profit_factor": m.profit_factor,
            "expectancy": m.expectancy, "max_drawdown": m.max_drawdown,
            "trade_count": m.trade_count, "rejected_signals": r.rejected_signals,
        }
    return summary


def stress_passes(results: Dict[str, SimulationResult],
                  *, min_pf: float = 1.0, max_dd: float = 0.5) -> bool:
    """Pass = every scenario remains viable (PF > 1 and DD bounded)."""
    for r in results.values():
        if r.metrics.profit_factor <= min_pf:
            return False
        if r.metrics.max_drawdown > max_dd:
            return False
    return True
