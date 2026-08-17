"""Optimizer: scores strategies by robust OOS metrics (not raw profit).

Ranking criteria (from spec section 42):
  OOS expectancy, OOS Profit Factor, max DD, walk-forward consistency,
  Monte Carlo survival, parameter stability, regime robustness, execution robustness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StrategyScore:
    name: str
    oos_expectancy: float
    oos_profit_factor: float
    max_drawdown: float
    wf_consistency: float  # fraction of passing windows
    mc_survival: float  # fraction of MC paths with positive terminal
    param_stability: float  # 0..1
    regime_robustness: float  # 0..1 (fraction of regimes profitable)
    execution_robustness: float  # 0..1
    composite: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)


# Weights (sum to 1.0). OOS performance dominates; historical profit is NOT used.
WEIGHTS = {
    "oos_expectancy": 0.22,
    "oos_profit_factor": 0.18,
    "max_drawdown": 0.12,
    "wf_consistency": 0.14,
    "mc_survival": 0.14,
    "param_stability": 0.08,
    "regime_robustness": 0.06,
    "execution_robustness": 0.06,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def _norm_expectancy(e: float) -> float:
    # map expectancy per trade into [0,1] with a soft saturating curve
    # assume typical scale is in R-multiples; 0R -> 0.5, +0.2R -> ~0.7, -0.2R -> ~0.3
    import math
    return 1.0 / (1.0 + math.exp(-5.0 * e))


def _norm_pf(pf: float) -> float:
    import math
    # PF 1.0 -> 0.5, 2.0 -> ~0.73, 0.5 -> ~0.27
    if pf <= 0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-(pf - 1.0)))


def _norm_dd(dd: float) -> float:
    # lower DD is better; map 0 DD -> 1.0, 50% DD -> 0.0
    return max(0.0, 1.0 - dd / 0.5)


def composite_score(s: StrategyScore) -> StrategyScore:
    comps = {
        "oos_expectancy": _norm_expectancy(s.oos_expectancy),
        "oos_profit_factor": _norm_pf(s.oos_profit_factor),
        "max_drawdown": _norm_dd(s.max_drawdown),
        "wf_consistency": max(0.0, min(1.0, s.wf_consistency)),
        "mc_survival": max(0.0, min(1.0, s.mc_survival)),
        "param_stability": max(0.0, min(1.0, s.param_stability)),
        "regime_robustness": max(0.0, min(1.0, s.regime_robustness)),
        "execution_robustness": max(0.0, min(1.0, s.execution_robustness)),
    }
    s.components = comps
    s.composite = sum(comps[k] * WEIGHTS[k] for k in WEIGHTS)
    return s


def rank_strategies(scores: List[StrategyScore]) -> List[StrategyScore]:
    for s in scores:
        composite_score(s)
    return sorted(scores, key=lambda s: s.composite, reverse=True)


def select_best(scores: List[StrategyScore]) -> Optional[StrategyScore]:
    ranked = rank_strategies(scores)
    return ranked[0] if ranked else None
