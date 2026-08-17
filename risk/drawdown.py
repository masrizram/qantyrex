"""Drawdown tracking: daily DD, emergency DD, dynamic risk reduction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..core.enums import SystemState
from ..core.exceptions import RiskLimitBreached


@dataclass
class DrawdownState:
    start_of_day_equity: float
    current_equity: float
    peak_equity: float

    @property
    def daily_dd(self) -> float:
        if self.start_of_day_equity <= 0:
            return 0.0
        return max(0.0, (self.start_of_day_equity - self.current_equity) / self.start_of_day_equity)

    @property
    def peak_dd(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)


@dataclass
class DrawdownConfig:
    daily_max_dd: float = 0.03
    emergency_dd: float = 0.05
    # Dynamic risk tiers: (dd_threshold, risk_factor)
    risk_tiers: tuple = (
        (0.00, 1.00),   # normal
        (0.015, 0.70),  # moderate DD
        (0.025, 0.50),  # high DD
        (0.03, 0.00),   # lock
    )


class DrawdownMonitor:
    def __init__(self, initial_equity: float,
                 daily_max_dd: float = 0.03,
                 emergency_dd: float = 0.05,
                 cfg: DrawdownConfig | None = None) -> None:
        self.cfg = cfg or DrawdownConfig(daily_max_dd=daily_max_dd, emergency_dd=emergency_dd)
        self.state = DrawdownState(
            start_of_day_equity=initial_equity,
            current_equity=initial_equity,
            peak_equity=initial_equity,
        )

    def update(self, equity: float) -> DrawdownState:
        if equity < 0:
            raise ValueError("Equity cannot be negative")
        self.state.current_equity = equity
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        return self.state

    def rollover_day(self, equity: Optional[float] = None) -> None:
        eq = equity if equity is not None else self.state.current_equity
        self.state.start_of_day_equity = eq
        if eq > self.state.peak_equity:
            self.state.peak_equity = eq

    def daily_dd_exceeded(self) -> bool:
        return self.state.daily_dd >= self.cfg.daily_max_dd

    def emergency_triggered(self) -> bool:
        return self.state.daily_dd >= self.cfg.emergency_dd

    def risk_factor(self) -> float:
        dd = self.state.daily_dd
        factor = 0.0
        for thr, f in self.cfg.risk_tiers:
            if dd >= thr:
                factor = f
        return factor

    def check(self) -> SystemState:
        """Return desired system state given current DD. Never silently bypasses."""
        if self.emergency_triggered():
            raise RiskLimitBreached(
                f"Emergency DD hit: {self.state.daily_dd:.4f} >= {self.cfg.emergency_dd}")
        if self.daily_dd_exceeded():
            return SystemState.RISK_LOCK
        return SystemState.RUNNING
