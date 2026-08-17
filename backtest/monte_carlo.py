"""Monte Carlo: resample trade sequence and apply random slippage/execution shocks.

Reports percentiles of terminal equity and drawdown, probability of ruin,
and max losing streak distribution. At least 10,000 iterations per the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class MonteCarloReport:
    iterations: int
    terminal_equity: np.ndarray  # shape (iterations,)
    max_drawdowns: np.ndarray   # shape (iterations,)
    max_losing_streaks: np.ndarray
    percentiles: dict  # 5,25,50,75,95 for return and DD
    probability_of_ruin: float
    median_return: float
    p5_return: float
    p95_return: float
    median_dd: float
    p95_dd: float


def monte_carlo(trades: pd.DataFrame, *,
                initial_equity: float = 10_000.0,
                iterations: int = 10_000,
                ruin_threshold: float = 0.5,  # ruin = equity drops to (1-ruin)*initial
                slippage_std_bps: float = 1.0,
                seed: int = 0) -> MonteCarloReport:
    """Resample the trade sequence (with replacement) and add per-trade
    slippage noise to model execution uncertainty."""
    if len(trades) == 0:
        return MonteCarloReport(
            iterations=0,
            terminal_equity=np.array([]), max_drawdowns=np.array([]),
            max_losing_streaks=np.array([]), percentiles={},
            probability_of_ruin=0.0, median_return=0.0,
            p5_return=0.0, p95_return=0.0, median_dd=0.0, p95_dd=0.0,
        )
    rng = np.random.default_rng(seed)
    pnls = trades["pnl"].astype(float).to_numpy()
    n = len(pnls)
    slip_std = slippage_std_bps / 10000.0 * initial_equity  # per-trade noise scale
    terminals = np.empty(iterations)
    dds = np.empty(iterations)
    streaks = np.empty(iterations, dtype=int)
    ruin_count = 0
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        sampled = pnls[idx]
        noise = rng.normal(0, slip_std, size=n)
        eq = initial_equity + np.cumsum(sampled + noise)
        terminals[i] = eq[-1]
        # drawdown
        running_max = np.maximum.accumulate(eq)
        dd = (running_max - eq) / running_max
        dds[i] = np.nanmax(np.where(np.isfinite(dd) & (running_max > 0), dd, 0.0))
        # max losing streak
        losing = (sampled + noise) < 0
        cur = best = 0
        for v in losing:
            if v: cur += 1; best = max(best, cur)
            else: cur = 0
        streaks[i] = best
        if eq.min() <= initial_equity * (1 - ruin_threshold):
            ruin_count += 1
    returns = (terminals - initial_equity) / initial_equity
    pct = {
        "return_5": float(np.percentile(returns, 5)),
        "return_25": float(np.percentile(returns, 25)),
        "return_50": float(np.percentile(returns, 50)),
        "return_75": float(np.percentile(returns, 75)),
        "return_95": float(np.percentile(returns, 95)),
        "dd_5": float(np.percentile(dds, 5)),
        "dd_25": float(np.percentile(dds, 25)),
        "dd_50": float(np.percentile(dds, 50)),
        "dd_75": float(np.percentile(dds, 75)),
        "dd_95": float(np.percentile(dds, 95)),
    }
    return MonteCarloReport(
        iterations=iterations, terminal_equity=terminals,
        max_drawdowns=dds, max_losing_streaks=streaks, percentiles=pct,
        probability_of_ruin=ruin_count / iterations,
        median_return=pct["return_50"], p5_return=pct["return_5"],
        p95_return=pct["return_95"], median_dd=pct["dd_50"], p95_dd=pct["dd_95"],
    )
