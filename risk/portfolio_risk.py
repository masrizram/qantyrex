"""Portfolio risk: aggregates drawdown + exposure + consecutive losses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..core.enums import SystemState
from ..core.exceptions import RiskLimitBreached
from .drawdown import DrawdownMonitor
from .exposure import ExposureMonitor, PositionExposure


@dataclass
class ConsecutiveLossState:
    streak: int = 0
    max_streak: int = 0
    max_consecutive_losses: int = 4

    def on_loss(self) -> int:
        self.streak += 1
        self.max_streak = max(self.max_streak, self.streak)
        return self.streak

    def on_win(self) -> None:
        self.streak = 0

    def exceeded(self) -> bool:
        return self.streak >= self.max_consecutive_losses


class PortfolioRisk:
    def __init__(self, equity: float, *, dd: DrawdownMonitor,
                 exposure: ExposureMonitor,
                 consec: ConsecutiveLossState) -> None:
        self.equity = equity
        self.dd = dd
        self.exposure = exposure
        self.consec = consec

    @classmethod
    def default(cls, equity: float,
                max_consecutive_losses: int = 4,
                daily_max_dd: float = 0.03,
                emergency_dd: float = 0.05) -> "PortfolioRisk":
        return cls(
            equity=equity,
            dd=DrawdownMonitor(equity),
            exposure=ExposureMonitor(equity),
            consec=ConsecutiveLossState(max_consecutive_losses=max_consecutive_losses),
        )

    def update_equity(self, equity: float) -> SystemState:
        self.equity = equity
        self.exposure.state.equity = equity
        self.dd.update(equity)
        return self.dd.check()  # may raise RiskLimitBreached on emergency

    def register_loss(self) -> SystemState:
        self.consec.on_loss()
        if self.consec.exceeded():
            return SystemState.RISK_LOCK
        return SystemState.RUNNING

    def register_win(self) -> None:
        self.consec.on_win()

    def set_positions(self, positions: List[PositionExposure]) -> List[str]:
        self.exposure.set_positions(positions)
        return self.exposure.violates()

    def risk_factor(self) -> float:
        """Combined risk multiplier in [0,1]. Loss-streak and DD both reduce it."""
        f_dd = self.dd.risk_factor()
        # reduce risk by 50% if near consecutive-loss limit
        f_streak = 0.5 if self.consec.streak >= max(1, self.consec.max_consecutive_losses - 1) else 1.0
        return min(f_dd, f_streak)

    def allowed_to_trade(self, open_positions: int, max_open: int) -> bool:
        if self.dd.daily_dd_exceeded():
            return False
        if self.consec.exceeded():
            return False
        if open_positions >= max_open:
            return False
        if self.exposure.violates():
            return False
        return True
