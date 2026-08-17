"""Realistic execution assumptions, clearly labeled.

If historical spread/tick/lot data is unavailable, a conservative model is used
and every assumption is tagged so reports never claim exact historical quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ExecutionAssumptions:
    fee_rate_taker: float = 0.001
    fee_rate_maker: float = 0.0006
    slippage_bps: float = 2.0
    spread_bps: float = 2.0  # conservative model when historical spread unavailable
    latency_ms: float = 500.0
    min_order_qty: float = 1e-6
    price_precision: int = 2
    qty_precision: int = 6
    lot_step: float = 1e-6
    tick_size: float = 0.01
    assumptions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.assumptions = {
            "fees": f"taker={self.fee_rate_taker} maker={self.fee_rate_maker} (configurable)",
            "slippage": f"{self.slippage_bps} bps (conservative model; no historical tick data)",
            "spread": f"{self.spread_bps} bps (conservative model; historical L2 unavailable)",
            "latency": f"{self.latency_ms} ms (assumed; no execution telemetry)",
            "min_order_qty": str(self.min_order_qty),
            "price_precision": str(self.price_precision),
            "qty_precision": str(self.qty_precision),
            "lot_step": str(self.lot_step),
            "tick_size": str(self.tick_size),
        }

    def label(self) -> str:
        return "ASSUMED_MODEL"  # not historical execution data

    def to_dict(self) -> dict:
        return {
            "fee_rate_taker": self.fee_rate_taker,
            "fee_rate_maker": self.fee_rate_maker,
            "slippage_bps": self.slippage_bps,
            "spread_bps": self.spread_bps,
            "latency_ms": self.latency_ms,
            "min_order_qty": self.min_order_qty,
            "price_precision": self.price_precision,
            "qty_precision": self.qty_precision,
            "lot_step": self.lot_step,
            "tick_size": self.tick_size,
            "label": self.label(),
            "assumptions": dict(self.assumptions),
        }


# Exchange-specific defaults (conservative). Caller can override per symbol.
EXCHANGE_DEFAULTS = {
    "binance": ExecutionAssumptions(fee_rate_taker=0.001, fee_rate_maker=0.00075,
                                    slippage_bps=2.0, spread_bps=2.0,
                                    min_order_qty=1e-6, price_precision=2, qty_precision=6),
    "gate": ExecutionAssumptions(fee_rate_taker=0.002, fee_rate_maker=0.0015,
                                  slippage_bps=3.0, spread_bps=4.0,
                                  min_order_qty=1e-8, price_precision=6, qty_precision=8),
    "bybit": ExecutionAssumptions(fee_rate_taker=0.001, fee_rate_maker=0.0006,
                                   slippage_bps=2.0, spread_bps=2.0,
                                   min_order_qty=1e-6, price_precision=2, qty_precision=6),
}


def for_exchange(name: str) -> ExecutionAssumptions:
    return EXCHANGE_DEFAULTS.get(name, ExecutionAssumptions())
