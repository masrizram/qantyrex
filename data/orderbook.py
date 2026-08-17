"""Order book snapshot model and basic spread/liquidity checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class OrderBookLevel:
    price: float
    amount: float


@dataclass
class OrderBookSnapshot:
    symbol: str
    timestamp: int
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid + self.best_ask) / 2
        return 0.0

    @property
    def spread(self) -> float:
        return max(0.0, self.best_ask - self.best_bid)

    @property
    def spread_percent(self) -> float:
        m = self.mid
        return (self.spread / m * 100.0) if m > 0 else 0.0

    def liquidity_at_distance(self, depth_pct: float = 0.5) -> float:
        """Sum of bid+ask volume within depth_pct of mid (liquidity proxy)."""
        m = self.mid
        if m == 0:
            return 0.0
        lo, hi = m * (1 - depth_pct / 100), m * (1 + depth_pct / 100)
        bid_vol = sum(b.amount for b in self.bids if lo <= b.price <= m)
        ask_vol = sum(a.amount for a in self.asks if m <= a.price <= hi)
        return bid_vol + ask_vol

    def is_valid(self, max_spread_percent: float) -> bool:
        return self.spread_percent <= max_spread_percent and self.mid > 0


def snapshot_from_dict(symbol: str, ts: int, raw: dict) -> OrderBookSnapshot:
    bids = [OrderBookLevel(float(p), float(a)) for p, a in raw.get("bids", [])]
    asks = [OrderBookLevel(float(p), float(a)) for p, a in raw.get("asks", [])]
    return OrderBookSnapshot(symbol=symbol, timestamp=ts, bids=bids, asks=asks)
