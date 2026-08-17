"""Performance metrics for trade results and equity curves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    net_profit: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    expectancy: float  # per-trade, in currency
    expectancy_r: float  # in R multiples
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    avg_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    recovery_factor: float
    max_losing_streak: int
    max_winning_streak: int
    trade_count: int
    total_fees: float
    total_slippage: float
    return_max_dd: float


def compute_metrics(trades: pd.DataFrame, equity_curve: Optional[pd.Series] = None,
                    risk_free: float = 0.0, periods_per_year: int = 252) -> Metrics:
    """`trades` must have columns: pnl, r_multiple, fees, slippage, exit_reason."""
    if len(trades) == 0:
        return Metrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0)
    pnls = trades["pnl"].astype(float).to_numpy()
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())  # positive number
    net = float(pnls.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    expectancy = float(pnls.mean())
    expectancy_r = float(trades["r_multiple"].astype(float).mean()) if "r_multiple" in trades else 0.0
    win_rate = float((pnls > 0).sum() / len(pnls))
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    total_fees = float(trades["fees"].astype(float).sum()) if "fees" in trades else 0.0
    total_slippage = float(trades["slippage"].astype(float).sum()) if "slippage" in trades else 0.0

    # streaks
    streaks_loss = _max_streak(pnls < 0)
    streaks_win = _max_streak(pnls > 0)

    # drawdown from equity curve or cumulative PnL
    if equity_curve is None:
        equity_curve = pd.Series(np.cumsum(np.concatenate([[0.0], pnls])))
    dd_series = _drawdown_series(equity_curve)
    max_dd = float(dd_series.max())
    avg_dd = float(dd_series.mean())

    # risk-adjusted returns: per-trade returns on equity
    eq = equity_curve.dropna()
    rets = eq.pct_change().dropna() if len(eq) > 1 else pd.Series(dtype=float)
    if len(rets) == 0:
        sharpe = 0.0
        sortino = 0.0
    else:
        r_std = rets.std()
        if not np.isfinite(r_std) or r_std == 0:
            sharpe = 0.0
        else:
            sharpe = float((rets.mean() - risk_free) / r_std * np.sqrt(periods_per_year))
        downside = rets[rets < 0]
        ds_std = downside.std() if len(downside) else 0.0
        if not np.isfinite(ds_std) or ds_std == 0:
            sortino = 0.0
        else:
            sortino = float((rets.mean() - risk_free) / ds_std * np.sqrt(periods_per_year))
    calmar = float(net / max_dd) if max_dd > 0 else 0.0
    recovery = float(net / max_dd) if max_dd > 0 else 0.0
    return_max_dd = float(net / max_dd) if max_dd > 0 else (float("inf") if net > 0 else 0.0)

    return Metrics(
        net_profit=net, gross_profit=gross_profit, gross_loss=gross_loss,
        profit_factor=pf, expectancy=expectancy, expectancy_r=expectancy_r,
        win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
        max_drawdown=max_dd, avg_drawdown=avg_dd, sharpe=sharpe, sortino=sortino,
        calmar=calmar, recovery_factor=recovery,
        max_losing_streak=streaks_loss, max_winning_streak=streaks_win,
        trade_count=len(trades), total_fees=total_fees, total_slippage=total_slippage,
        return_max_dd=return_max_dd,
    )


def _max_streak(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    equity = equity.replace(0, np.nan)
    running_max = equity.cummax()
    dd = (running_max - equity) / running_max
    return dd.fillna(0.0)


def metrics_to_dict(m: Metrics) -> dict:
    return {
        "net_profit": m.net_profit, "gross_profit": m.gross_profit,
        "gross_loss": m.gross_loss, "profit_factor": m.profit_factor,
        "expectancy": m.expectancy, "expectancy_r": m.expectancy_r,
        "win_rate": m.win_rate, "avg_win": m.avg_win, "avg_loss": m.avg_loss,
        "max_drawdown": m.max_drawdown, "avg_drawdown": m.avg_drawdown,
        "sharpe": m.sharpe, "sortino": m.sortino, "calmar": m.calmar,
        "recovery_factor": m.recovery_factor,
        "max_losing_streak": m.max_losing_streak,
        "max_winning_streak": m.max_winning_streak,
        "trade_count": m.trade_count, "total_fees": m.total_fees,
        "total_slippage": m.total_slippage, "return_max_dd": m.return_max_dd,
    }
