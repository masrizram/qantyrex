"""Live performance tracking: rolling expectancy, PF, DD, win rate, slippage."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import numpy as np


@dataclass
class RollingPerformance:
    window: int
    trades: Deque[float] = field(default_factory=lambda: deque())
    pnls: Deque[float] = field(default_factory=lambda: deque())
    slippages: Deque[float] = field(default_factory=lambda: deque())
    spreads: Deque[float] = field(default_factory=lambda: deque())

    def add_trade(self, pnl: float, slippage: float = 0.0, spread: float = 0.0) -> None:
        self.trades.append(pnl)
        self.pnls.append(pnl)
        self.slippages.append(slippage)
        self.spreads.append(spread)
        while len(self.trades) > self.window:
            self.trades.popleft(); self.pnls.popleft()
            self.slippages.popleft(); self.spreads.popleft()

    @property
    def expectancy(self) -> float:
        return float(np.mean(self.pnls)) if self.pnls else 0.0

    @property
    def win_rate(self) -> float:
        if not self.pnls:
            return 0.0
        return sum(1 for p in self.pnls if p > 0) / len(self.pnls)

    @property
    def profit_factor(self) -> float:
        pos = sum(p for p in self.pnls if p > 0)
        neg = -sum(p for p in self.pnls if p < 0)
        return pos / neg if neg > 0 else (float("inf") if pos > 0 else 0.0)

    @property
    def avg_slippage(self) -> float:
        return float(np.mean(self.slippages)) if self.slippages else 0.0

    @property
    def avg_spread(self) -> float:
        return float(np.mean(self.spreads)) if self.spreads else 0.0

    @property
    def drawdown(self) -> float:
        if not self.pnls:
            return 0.0
        eq = np.cumsum(list(self.pnls))
        running = np.maximum.accumulate(eq)
        dd = (running - eq)
        # normalize by initial equity proxy = peak
        return float(np.max(dd) / max(running.max(), 1e-9)) if len(eq) else 0.0

    def snapshot(self) -> Dict[str, float]:
        return {
            "expectancy": self.expectancy, "win_rate": self.win_rate,
            "profit_factor": self.profit_factor, "avg_slippage": self.avg_slippage,
            "avg_spread": self.avg_spread, "drawdown": self.drawdown,
            "n_trades": len(self.pnls),
        }
