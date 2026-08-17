"""Top-level risk manager: ties sizing, drawdown, exposure, breakers together.

This is the single entry point the rest of the system calls before opening or
modifying a position. It is fail-closed: any risk violation returns a denial.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..config import Config
from ..core.enums import Side, SystemState
from ..core.exceptions import RiskLimitBreached
from ..core.models import Signal
from .circuit_breaker import CircuitBreaker
from .drawdown import DrawdownMonitor
from .exposure import ExposureMonitor, PositionExposure
from .portfolio_risk import PortfolioRisk, ConsecutiveLossState
from .position_sizer import SizingInputs, SizingResult, size_position


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    sizing: Optional[SizingResult] = None
    desired_state: Optional[SystemState] = None


class RiskManager:
    def __init__(self, cfg: Config, equity: float) -> None:
        self.cfg = cfg
        self.equity = equity
        self.dd = DrawdownMonitor(
            equity,
            daily_max_dd=cfg.daily_max_drawdown,
            emergency_dd=cfg.emergency_drawdown,
        )
        self.exposure = ExposureMonitor(equity)
        self.consec = ConsecutiveLossState(max_consecutive_losses=cfg.max_consecutive_losses)
        self.portfolio = PortfolioRisk(equity, dd=self.dd, exposure=self.exposure,
                                       consec=self.consec)
        self.breakers = CircuitBreaker()
        self._register_breakers()

    def _register_breakers(self) -> None:
        self.breakers.register("daily_dd", lambda: (
            not self.dd.daily_dd_exceeded(), "daily_dd_exceeded"))
        self.breakers.register("emergency_dd", lambda: (
            not self.dd.emergency_triggered(), "emergency_dd"))
        self.breakers.register("consecutive_losses", lambda: (
            not self.consec.exceeded(), "consecutive_losses_exceeded"))

    def update_equity(self, equity: float) -> SystemState:
        self.equity = equity
        return self.portfolio.update_equity(equity)

    def evaluate_signal(self, signal: Signal, open_positions: int,
                        open_position_notional: float = 0.0) -> RiskDecision:
        # 1. State check
        desired = self.breakers.desired_state()
        if desired in (SystemState.RISK_LOCK, SystemState.SHUTDOWN):
            return RiskDecision(False, f"breaker_state_{desired.value}", desired_state=desired)
        # 2. Position count
        if open_positions >= self.cfg.max_open_positions:
            return RiskDecision(False, "max_open_positions", desired_state=desired)
        # 3. Dynamic risk factor (DD + consecutive losses)
        factor = self.portfolio.risk_factor()
        if factor <= 0:
            return RiskDecision(False, "risk_factor_zero", desired_state=desired)
        effective_risk = min(self.cfg.initial_risk_percent * factor,
                             self.cfg.max_risk_percent)
        if effective_risk <= 0:
            return RiskDecision(False, "effective_risk_zero", desired_state=desired)
        # 4. Position sizing
        sin = SizingInputs(
            equity=self.equity,
            risk_percent=effective_risk,
            entry=signal.entry, stop_loss=signal.stop_loss,
            fee_rate=self.cfg.fee_rate, slippage_bps=self.cfg.slippage_bps,
            min_qty=1e-6, min_notional=0.0,
        )
        sizing = size_position(sin, self.cfg.max_risk_percent)
        if sizing.quantity <= 0:
            return RiskDecision(False, f"sizing_rejected:{sizing.reason}",
                                sizing=sizing, desired_state=desired)
        # 5. Exposure check (post-trade hypothetical)
        new_pos = PositionExposure(
            symbol=signal.symbol, side=signal.side.value,
            notional=sizing.notional + open_position_notional,
        )
        existing = list(self.exposure.state.positions.values())
        self.exposure.set_positions(existing + [new_pos])
        violations = self.exposure.violates()
        # restore prior exposure state
        self.exposure.set_positions(existing)
        if violations:
            return RiskDecision(False, f"exposure_violation:{violations[0]}",
                                sizing=sizing, desired_state=desired)
        return RiskDecision(True, "ok", sizing=sizing, desired_state=desired)

    def on_trade_closed(self, pnl: float) -> SystemState:
        if pnl < 0:
            self.consec.on_loss()
        else:
            self.consec.on_win()
        new_equity = self.equity + pnl
        self.update_equity(new_equity)
        if self.consec.exceeded():
            return SystemState.RISK_LOCK
        return self.dd.check()
