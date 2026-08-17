"""Position sizing: risk-based, with exchange/contract constraints.

Risk capital = equity * risk_percent
Price risk  = abs(entry - SL)
Raw size    = risk_capital / price_risk

Then constrained by: lot/tick size, exchange minimum notional, fees, slippage,
leverage, max position size. Final expected loss must remain within risk budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math


@dataclass
class SizingInputs:
    equity: float
    risk_percent: float
    entry: float
    stop_loss: float
    fee_rate: float = 0.0
    slippage_bps: float = 0.0
    contract_size: float = 1.0
    tick_size: float = 0.01
    lot_step: float = 0.0001
    min_qty: float = 0.0001
    min_notional: float = 0.0
    leverage: float = 1.0
    max_qty: float = float("inf")


@dataclass
class SizingResult:
    quantity: float
    notional: float
    expected_loss: float
    risk_capital: float
    price_risk: float
    constrained: bool
    reason: Optional[str] = None


def size_position(inp: SizingInputs, max_risk_percent: float) -> SizingResult:
    if inp.risk_percent <= 0:
        return SizingResult(0, 0, 0, 0, 0, True, "risk_percent_zero")
    if inp.risk_percent > max_risk_percent:
        return SizingResult(0, 0, 0, 0, 0, True,
                            f"risk_percent {inp.risk_percent} > max {max_risk_percent}")

    risk_capital = inp.equity * inp.risk_percent
    price_risk = abs(inp.entry - inp.stop_loss)
    if price_risk <= 0:
        return SizingResult(0, 0, 0, risk_capital, price_risk, True, "zero_price_risk")

    raw_qty = risk_capital / price_risk

    # transaction costs per unit: fees on entry+exit, slippage on entry+exit
    # cost_per_unit = fee_rate*(2*entry) + slippage*(2*entry)
    slip = inp.slippage_bps / 10000.0
    cost_per_unit = inp.entry * (2 * (inp.fee_rate + slip))
    # subtract cost from risk budget to ensure final expected loss within budget
    # expected_loss = qty * (price_risk + cost_per_unit) <= risk_capital
    denom = price_risk + cost_per_unit
    if denom <= 0:
        return SizingResult(0, 0, 0, risk_capital, price_risk, True, "nonpositive_denom")
    cost_aware_qty = risk_capital / denom

    qty = min(raw_qty, cost_aware_qty)

    # Contract / lot step rounding (floor to lot_step)
    if inp.lot_step > 0:
        qty = math.floor(qty / inp.lot_step) * inp.lot_step

    # Exchange min qty
    if qty < inp.min_qty:
        return SizingResult(0, 0, 0, risk_capital, price_risk, True,
                            f"qty {qty} < min_qty {inp.min_qty}")
    notional = qty * inp.entry
    if inp.min_notional > 0 and notional < inp.min_notional:
        return SizingResult(0, 0, 0, risk_capital, price_risk, True,
                            f"notional {notional} < min_notional {inp.min_notional}")

    # Max qty (from leverage / exchange cap)
    if qty > inp.max_qty:
        qty = inp.max_qty
        if inp.lot_step > 0:
            qty = math.floor(qty / inp.lot_step) * inp.lot_step

    expected_loss = qty * (price_risk + cost_per_unit)
    constrained = expected_loss > risk_capital * 1.0001  # tiny float tolerance
    return SizingResult(
        quantity=qty, notional=notional, expected_loss=expected_loss,
        risk_capital=risk_capital, price_risk=price_risk,
        constrained=constrained,
        reason="ok" if not constrained else "expected_loss_exceeds_budget",
    )
