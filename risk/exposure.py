"""Exposure tracking: directional, symbol, margin, leverage, correlation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PositionExposure:
    symbol: str
    side: str  # BUY/SELL
    notional: float
    weight: float = 0.0  # notional / equity


@dataclass
class ExposureState:
    equity: float = 0.0
    positions: Dict[str, PositionExposure] = field(default_factory=dict)
    correlation_matrix: Optional[np.ndarray] = None  # symbols x symbols
    symbols: List[str] = field(default_factory=list)
    margin_used: float = 0.0

    @property
    def total_exposure(self) -> float:
        return sum(p.notional for p in self.positions.values())

    @property
    def net_directional(self) -> float:
        net = 0.0
        for p in self.positions.values():
            sign = 1.0 if p.side == "BUY" else -1.0
            net += sign * p.notional
        return net

    @property
    def gross_leverage(self) -> float:
        return self.total_exposure / self.equity if self.equity > 0 else 0.0


@dataclass
class ExposureConfig:
    max_gross_leverage: float = 1.0
    max_single_symbol_weight: float = 1.0
    max_correlated_weight: float = 0.6  # for a correlated cluster
    correlation_threshold: float = 0.7


class ExposureMonitor:
    def __init__(self, equity: float, cfg: ExposureConfig | None = None) -> None:
        self.cfg = cfg or ExposureConfig()
        self.state = ExposureState(equity=equity)

    def set_positions(self, positions: List[PositionExposure]) -> None:
        self.state.positions = {p.symbol: p for p in positions}
        for p in self.state.positions.values():
            p.weight = p.notional / self.state.equity if self.state.equity > 0 else 0.0

    def set_correlation(self, symbols: List[str], matrix: np.ndarray) -> None:
        if matrix.shape != (len(symbols), len(symbols)):
            raise ValueError("Correlation matrix shape mismatch")
        self.state.symbols = symbols
        self.state.correlation_matrix = matrix

    def violates(self) -> List[str]:
        reasons: List[str] = []
        if self.state.gross_leverage > self.cfg.max_gross_leverage:
            reasons.append(f"gross_leverage {self.state.gross_leverage:.2f} > max")
        for sym, p in self.state.positions.items():
            if p.weight > self.cfg.max_single_symbol_weight:
                reasons.append(f"{sym} weight {p.weight:.2f} > max")
        # correlation cluster check
        if self.state.correlation_matrix is not None and len(self.state.symbols) > 1:
            sym_idx = {s: i for i, s in enumerate(self.state.symbols)}
            # Union-find style clustering: group symbols whose |corr| >= threshold
            n = len(self.state.symbols)
            parent = list(range(n))
            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]; a = parent[a]
                return a
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(self.state.correlation_matrix[i, j]) >= self.cfg.correlation_threshold:
                        parent[find(i)] = find(j)
            clusters: Dict[int, List[str]] = {}
            for i, s in enumerate(self.state.symbols):
                clusters.setdefault(find(i), []).append(s)
            for members in clusters.values():
                if len(members) < 2:
                    continue
                cluster_weight = sum(
                    self.state.positions[m].weight for m in members
                    if m in self.state.positions
                )
                if cluster_weight > self.cfg.max_correlated_weight:
                    reasons.append(f"correlated cluster weight {cluster_weight:.2f} > max")
        return reasons
