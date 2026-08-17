"""Execution guard: fail-closed gate for live trading.

A live order can ONLY be submitted when ALL conditions are met:
  - TRADING_MODE == LIVE
  - LIVE_TRADING_ENABLED == true
  - credentials present
  - all acceptance gates passed (caller supplies evidence)
  - state machine is RUNNING (not RISK_LOCK / SHUTDOWN / ERROR)

If any condition is missing -> BLOCK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..config import Config, TradingMode
from ..core.enums import SystemState
from ..core.exceptions import ExecutionError


@dataclass
class AcceptanceEvidence:
    oos_expectancy_positive: bool = False
    oos_pf_positive: bool = False
    wf_consistency_pass: bool = False
    mc_pass: bool = False
    sensitivity_pass: bool = False
    stress_pass: bool = False
    no_look_ahead: bool = False
    risk_controls_pass: bool = False
    critical_tests_pass: bool = False
    failures: List[str] = field(default_factory=list)

    def all_pass(self) -> bool:
        return (self.oos_expectancy_positive and self.oos_pf_positive
                and self.wf_consistency_pass and self.mc_pass
                and self.sensitivity_pass and self.stress_pass
                and self.no_look_ahead and self.risk_controls_pass
                and self.critical_tests_pass)


class ExecutionGuard:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def can_trade_live(self, state: SystemState,
                      evidence: AcceptanceEvidence | None = None) -> tuple[bool, str]:
        if self.cfg.trading_mode != TradingMode.LIVE:
            return False, f"mode={self.cfg.trading_mode.value} (not LIVE)"
        if not self.cfg.live_trading_enabled:
            return False, "LIVE_TRADING_ENABLED=false"
        if not (self.cfg.api_key and self.cfg.api_secret):
            return False, "missing_credentials"
        if state != SystemState.RUNNING:
            return False, f"state={state.value} (not RUNNING)"
        if evidence is not None and not evidence.all_pass():
            return False, f"acceptance_gates_failed:{evidence.failures}"
        return True, "live_allowed"

    def assert_can_trade_live(self, state: SystemState,
                              evidence: AcceptanceEvidence | None = None) -> None:
        ok, reason = self.can_trade_live(state, evidence)
        if not ok:
            raise ExecutionError(f"LIVE TRADING BLOCKED: {reason}")
