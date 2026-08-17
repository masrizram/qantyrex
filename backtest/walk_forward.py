"""Walk-forward analysis: rolling train/validate/OOS, aggregate all OOS windows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import pandas as pd

from .engine import Backtester, BacktestResult


@dataclass
class WalkForwardWindow:
    window_index: int
    train_start: int
    train_end: int
    oos_start: int
    oos_end: int
    result: BacktestResult


@dataclass
class WalkForwardReport:
    windows: List[WalkForwardWindow] = field(default_factory=list)
    total_oos_trades: int = 0
    passing_windows: int = 0
    failing_windows: int = 0
    consistency: float = 0.0  # passing / total
    aggregate_oos_expectancy: float = 0.0
    aggregate_oos_pf: float = 0.0


def walk_forward(
    df: pd.DataFrame,
    runner: Callable[[pd.DataFrame, str], BacktestResult],
    *,
    train_size: int = 400,
    oos_size: int = 100,
    step: int = 100,
    min_oos_expectancy: float = 0.0,
    min_oos_pf: float = 1.0,
) -> WalkForwardReport:
    """Roll windows over `df`. Each window: train on [s, s+train), OOS on next oos_size bars.

    The `runner` is invoked with (window_df, split_label) and must return a
    BacktestResult. We aggregate EVERY OOS period's metrics — no cherry-picking.
    """
    report = WalkForwardReport()
    n = len(df)
    start = 0
    widx = 0
    ts_col = df["timestamp"].to_numpy() if "timestamp" in df.columns else None
    while start + train_size + oos_size <= n:
        train_df = df.iloc[start:start + train_size]
        oos_df = df.iloc[start + train_size:start + train_size + oos_size]
        result = runner(oos_df, f"OOS_WIN{widx}")
        passing = (result.result.metrics.expectancy > min_oos_expectancy
                   and result.result.metrics.profit_factor > min_oos_pf
                   and result.result.metrics.trade_count > 0)
        # record positional indices to avoid assuming an int index
        train_start_v = int(train_df.index[0]) if ts_col is None else int(ts_col[start])
        train_end_v = int(train_df.index[-1]) if ts_col is None else int(ts_col[start + train_size - 1])
        oos_start_v = int(oos_df.index[0]) if ts_col is None else int(ts_col[start + train_size])
        oos_end_v = int(oos_df.index[-1]) if ts_col is None else int(ts_col[start + train_size + oos_size - 1])
        wf = WalkForwardWindow(
            window_index=widx, train_start=train_start_v, train_end=train_end_v,
            oos_start=oos_start_v, oos_end=oos_end_v,
            result=result,
        )
        report.windows.append(wf)
        report.total_oos_trades += result.result.metrics.trade_count
        if passing:
            report.passing_windows += 1
        else:
            report.failing_windows += 1
        start += step
        widx += 1

    total = report.passing_windows + report.failing_windows
    report.consistency = (report.passing_windows / total) if total > 0 else 0.0
    # aggregate expectancy/PF across OOS windows (trade-weighted)
    num = 0.0; den = 0.0; exp_num = 0.0; exp_den = 0.0
    for w in report.windows:
        m = w.result.result.metrics
        if m.gross_loss > 0:
            num += m.gross_profit
            den += m.gross_loss
        exp_num += m.expectancy * m.trade_count
        exp_den += m.trade_count
    report.aggregate_oos_pf = (num / den) if den > 0 else (
        float("inf") if num > 0 else 0.0)
    report.aggregate_oos_expectancy = (exp_num / exp_den) if exp_den > 0 else 0.0
    return report
