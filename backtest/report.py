"""Report generation: final performance + scorecard + live readiness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from ..core.enums import LiveReadiness
from .metrics import Metrics, metrics_to_dict
from .walk_forward import WalkForwardReport
from .monte_carlo import MonteCarloReport
from .stress_test import stress_summary
from ..research.optimizer import StrategyScore


def performance_report(
    *,
    strategy_name: str,
    backtest_metrics: Metrics,
    oos_metrics: Metrics,
    wf: WalkForwardReport,
    mc: MonteCarloReport,
    stress_summary: dict,
    execution_summary: dict,
    risk_summary: dict,
    readiness: LiveReadiness,
    statistical_edge: bool,
    robustness: float,
) -> str:
    lines = []
    bm = metrics_to_dict(backtest_metrics)
    om = metrics_to_dict(oos_metrics)
    lines.append("=" * 60)
    lines.append("FINAL PERFORMANCE REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("STRATEGY")
    lines.append(f"  Name                 {strategy_name}")
    lines.append("")
    lines.append("BACKTEST")
    lines.append(f"  Net Profit           {bm['net_profit']:.2f}")
    lines.append(f"  Profit Factor        {bm['profit_factor']:.3f}")
    lines.append(f"  Expectancy           {bm['expectancy']:.4f}")
    lines.append(f"  Win Rate             {bm['win_rate']:.3f}")
    lines.append(f"  Max Drawdown         {bm['max_drawdown']:.3f}")
    lines.append(f"  Sharpe               {bm['sharpe']:.3f}")
    lines.append(f"  Sortino              {bm['sortino']:.3f}")
    lines.append(f"  Trade Count          {bm['trade_count']}")
    lines.append("")
    lines.append("OOS")
    lines.append(f"  OOS Net Profit       {om['net_profit']:.2f}")
    lines.append(f"  OOS Profit Factor    {om['profit_factor']:.3f}")
    lines.append(f"  OOS Expectancy       {om['expectancy']:.4f}")
    lines.append(f"  OOS Drawdown         {om['max_drawdown']:.3f}")
    lines.append("")
    lines.append("WALK-FORWARD")
    lines.append(f"  Windows              {len(wf.windows)}")
    lines.append(f"  Passing              {wf.passing_windows}")
    lines.append(f"  Failing              {wf.failing_windows}")
    lines.append(f"  Consistency          {wf.consistency:.3f}")
    lines.append("")
    lines.append("MONTE CARLO")
    lines.append(f"  Iterations           {mc.iterations}")
    lines.append(f"  Median Return        {mc.median_return:.4f}")
    lines.append(f"  5th Percentile       {mc.p5_return:.4f}")
    lines.append(f"  95th Percentile      {mc.p95_return:.4f}")
    lines.append(f"  Median DD            {mc.median_dd:.4f}")
    lines.append(f"  95th Percentile DD   {mc.p95_dd:.4f}")
    lines.append(f"  Probability of Ruin  {mc.probability_of_ruin:.4f}")
    lines.append("")
    lines.append("EXECUTION")
    for k, v in execution_summary.items():
        lines.append(f"  {k:<20} {v}")
    lines.append("")
    lines.append("RISK")
    for k, v in risk_summary.items():
        lines.append(f"  {k:<20} {v}")
    lines.append("")
    lines.append("STRESS")
    for name, m in stress_summary.items():
        lines.append(f"  {name:<20} PF={m['profit_factor']:.3f} DD={m['max_drawdown']:.3f} trades={m['trade_count']}")
    lines.append("")
    lines.append("FINAL")
    lines.append(f"  Strategy Status      {readiness.value}")
    lines.append(f"  Statistical Edge     {'YES' if statistical_edge else 'NO'}")
    lines.append(f"  Robustness           {robustness:.3f}")
    lines.append("")
    return "\n".join(lines)


def scorecard(scores: dict) -> str:
    lines = []
    lines.append("=" * 50)
    lines.append("QUANT SYSTEM SCORECARD")
    lines.append("=" * 50)
    lines.append("")
    total = 0
    for k, v in scores.items():
        lines.append(f"{k:<26} {v:>2}/10")
        total += v
    lines.append("")
    lines.append(f"{'TOTAL':<26} {total}/200")
    lines.append("=" * 50)
    return "\n".join(lines)
