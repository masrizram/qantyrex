"""Reconciliation: compare local state with exchange state, fail-closed on mismatch."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..core.enums import SystemState
from ..core.exceptions import ReconciliationMismatch
from ..core.models import Position


@dataclass
class ReconciliationReport:
    unknown_positions: List[Dict] = field(default_factory=list)
    missing_positions: List[Dict] = field(default_factory=list)
    incorrect_size: List[Dict] = field(default_factory=list)
    missing_sl: List[str] = field(default_factory=list)
    missing_tp: List[str] = field(default_factory=list)
    stale_orders: List[str] = field(default_factory=list)
    duplicate_orders: List[str] = field(default_factory=list)

    def has_issues(self) -> bool:
        return any([
            self.unknown_positions, self.missing_positions, self.incorrect_size,
            self.missing_sl, self.missing_tp, self.stale_orders, self.duplicate_orders,
        ])

    def summary(self) -> List[str]:
        out = []
        if self.unknown_positions:
            out.append(f"unknown_positions:{len(self.unknown_positions)}")
        if self.missing_positions:
            out.append(f"missing_positions:{len(self.missing_positions)}")
        if self.incorrect_size:
            out.append(f"incorrect_size:{len(self.incorrect_size)}")
        if self.missing_sl:
            out.append(f"missing_sl:{len(self.missing_sl)}")
        if self.missing_tp:
            out.append(f"missing_tp:{len(self.missing_tp)}")
        if self.stale_orders:
            out.append(f"stale_orders:{len(self.stale_orders)}")
        if self.duplicate_orders:
            out.append(f"duplicate_orders:{len(self.duplicate_orders)}")
        return out


def reconcile(
    local_positions: List[Position],
    exchange_positions: List[Dict],
    local_orders: List[Dict],
    exchange_orders: List[Dict],
    *,
    size_tolerance: float = 0.0001,
) -> ReconciliationReport:
    """Compare local vs exchange. Any unresolved mismatch raises ReconciliationMismatch
    (caller decides whether to risk-lock; this function reports all issues)."""
    rep = ReconciliationReport()
    local_by_sym = {p.symbol: p for p in local_positions if p.is_open}
    exch_by_sym = {p.get("symbol"): p for p in exchange_positions}

    for sym, p in exch_by_sym.items():
        if sym not in local_by_sym:
            rep.unknown_positions.append(p)
    for sym, p in local_by_sym.items():
        if sym not in exch_by_sym:
            rep.missing_positions.append({"symbol": sym, "side": p.side.value,
                                           "qty": p.quantity})
        else:
            ex_qty = float(exch_by_sym[sym].get("contracts")
                            or exch_by_sym[sym].get("qty")
                            or 0)
            if abs(ex_qty - p.quantity) > size_tolerance:
                rep.incorrect_size.append({"symbol": sym, "local": p.quantity, "exchange": ex_qty})
        # missing SL/TP
        if p.is_open:
            if p.stop_loss is None or p.stop_loss <= 0:
                rep.missing_sl.append(sym)
            if p.take_profit is None or p.take_profit <= 0:
                rep.missing_tp.append(sym)

    # duplicate/stale orders
    seen_ids = set()
    for o in local_orders:
        oid = o.get("client_order_id") or o.get("id")
        if oid in seen_ids:
            rep.duplicate_orders.append(oid)
        seen_ids.add(oid)
    exch_order_ids = {o.get("id") for o in exchange_orders}
    for o in local_orders:
        if o.get("status") == "SUBMITTED" and o.get("id") and o.get("id") not in exch_order_ids:
            # may be stale if exchange no longer knows about it
            rep.stale_orders.append(o.get("id"))
    return rep


def raise_on_critical(rep: ReconciliationReport) -> None:
    """Any unresolved mismatch -> raise (fail-closed)."""
    if rep.has_issues():
        raise ReconciliationMismatch(
            f"Reconciliation mismatch: {rep.summary()}")
