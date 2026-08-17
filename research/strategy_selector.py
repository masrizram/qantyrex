"""Strategy selector: promote/reject based on acceptance gates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..core.enums import LiveReadiness
from .optimizer import StrategyScore


@dataclass
class PromotionDecision:
    readiness: LiveReadiness
    reasons: List[str] = field(default_factory=list)
    score: Optional[StrategyScore] = None


def evaluate_readiness(s: StrategyScore, *,
                       min_oos_expectancy: float = 0.0,
                       min_oos_pf: float = 1.0,
                       max_drawdown: float = 0.25,
                       min_wf_consistency: float = 0.6,
                       min_mc_survival: float = 0.85,
                       min_composite: float = 0.6,
                       paper_verified: bool = False,
                       micro_live_verified: bool = False) -> PromotionDecision:
    reasons: List[str] = []

    # Hard gates (spec section 56)
    if s.oos_expectancy <= min_oos_expectancy:
        reasons.append(f"oos_expectancy {s.oos_expectancy:.4f} <= {min_oos_expectancy}")
    if s.oos_profit_factor <= min_oos_pf:
        reasons.append(f"oos_pf {s.oos_profit_factor:.3f} <= {min_oos_pf}")
    if s.max_drawdown > max_drawdown:
        reasons.append(f"max_dd {s.max_drawdown:.3f} > {max_drawdown}")
    if s.wf_consistency < min_wf_consistency:
        reasons.append(f"wf_consistency {s.wf_consistency:.2f} < {min_wf_consistency}")
    if s.mc_survival < min_mc_survival:
        reasons.append(f"mc_survival {s.mc_survival:.2f} < {min_mc_survival}")

    hard_gates_ok = len(reasons) == 0

    # Composite is informational only, but we surface it
    if s.composite < min_composite:
        reasons.append(f"composite {s.composite:.3f} < {min_composite}")

    if not hard_gates_ok:
        return PromotionDecision(LiveReadiness.REJECTED, reasons, s)

    # Stage gating: don't skip stages
    if micro_live_verified:
        return PromotionDecision(LiveReadiness.PRODUCTION_CANDIDATE, reasons, s)
    if paper_verified:
        return PromotionDecision(LiveReadiness.MICRO_LIVE, reasons, s)
    if s.oos_expectancy > 0 and s.oos_profit_factor > 1.0:
        return PromotionDecision(LiveReadiness.PAPER_TRADING, reasons, s)
    return PromotionDecision(LiveReadiness.BACKTEST_VERIFIED, reasons, s)
