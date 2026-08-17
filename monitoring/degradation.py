"""Strategy degradation detection: compare live rolling metrics vs OOS baseline.

If degradation exceeds thresholds -> REDUCE RISK. If severe -> RISK_LOCK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from ..core.enums import SystemState
from .performance import RollingPerformance


@dataclass
class DegradationConfig:
    oos_expectancy: float = 0.0
    oos_profit_factor: float = 1.0
    min_trades_to_evaluate: int = 20
    reduce_threshold: float = 0.5   # if live < 50% of OOS -> reduce
    lock_threshold: float = 0.0     # if live expectancy <= 0 -> lock


@dataclass
class DegradationStatus:
    degraded: bool = False
    severe: bool = False
    reason: str = ""
    live_expectancy: float = 0.0
    live_pf: float = 0.0
    risk_factor: float = 1.0


class DegradationMonitor:
    def __init__(self, cfg: DegradationConfig,
                 perf: Optional[RollingPerformance] = None) -> None:
        self.cfg = cfg
        self.perf = perf or RollingPerformance(window=50)

    def add_trade(self, pnl: float, slippage: float = 0.0, spread: float = 0.0) -> None:
        self.perf.add_trade(pnl, slippage, spread)

    def evaluate(self) -> DegradationStatus:
        if len(self.perf.pnls) < self.cfg.min_trades_to_evaluate:
            return DegradationStatus(risk_factor=1.0,
                                      reason=f"insufficient_trades ({len(self.perf.pnls)})")
        live_e = self.perf.expectancy
        live_pf = self.perf.profit_factor
        if live_e <= self.cfg.lock_threshold:
            return DegradationStatus(degraded=True, severe=True,
                                     reason="live_expectancy_non_positive",
                                     live_expectancy=live_e, live_pf=live_pf,
                                     risk_factor=0.0)
        if self.cfg.oos_expectancy > 0 and live_e < self.cfg.reduce_threshold * self.cfg.oos_expectancy:
            factor = max(0.0, live_e / max(self.cfg.oos_expectancy, 1e-9))
            return DegradationStatus(degraded=True, severe=False,
                                     reason="expectancy_below_oos_fraction",
                                     live_expectancy=live_e, live_pf=live_pf,
                                     risk_factor=factor)
        if self.cfg.oos_profit_factor > 1.0 and live_pf < 1.0:
            return DegradationStatus(degraded=True, severe=False,
                                     reason="pf_below_one",
                                     live_expectancy=live_e, live_pf=live_pf,
                                     risk_factor=0.5)
        return DegradationStatus(degraded=False, severe=False, reason="ok",
                                 live_expectancy=live_e, live_pf=live_pf,
                                 risk_factor=1.0)
