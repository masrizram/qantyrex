"""Order manager: idempotency, lifecycle, duplicate prevention.

Every order carries signal_id, trade_id, client_order_id, strategy_version.
We maintain a registry of seen client_order_ids to prevent duplicate submission
of the same signal (idempotency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.enums import OrderStatus, OrderType, Side
from ..core.exceptions import ExecutionError, OrderRejected
from ..core.models import Order, Signal
from .executor import BaseExecutor


class OrderManager:
    def __init__(self, executor: BaseExecutor) -> None:
        self.executor = executor
        self._seen_client_ids: Dict[str, Order] = {}
        self._open_orders: Dict[str, Order] = {}
        self._all_orders: List[Order] = []

    async def submit(self, signal: Signal, quantity: float,
                     order_type: OrderType = OrderType.MARKET,
                     price: Optional[float] = None) -> Order:
        # Idempotency: same signal_id -> return the previously submitted order
        for cid, o in self._seen_client_ids.items():
            if o.signal_id == signal.signal_id and o.status in (
                    OrderStatus.SUBMITTED, OrderStatus.FILLED, OrderStatus.PARTIAL):
                return o
        order = Order(
            signal_id=signal.signal_id, trade_id=None,
            strategy_version=signal.strategy_version, symbol=signal.symbol,
            side=signal.side, order_type=order_type, quantity=quantity,
            price=price,
        )
        # pre-register client_order_id for idempotency
        self._seen_client_ids[order.client_order_id] = order
        filled = await self.executor.create_order(order)
        self._all_orders.append(filled)
        if filled.status == OrderStatus.FILLED:
            self._open_orders.pop(filled.client_order_id, None)
        elif filled.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
            self._open_orders[filled.client_order_id] = filled
        elif filled.status == OrderStatus.REJECTED:
            raise OrderRejected(filled.reject_reason or "rejected")
        return filled

    async def cancel(self, order: Order) -> Order:
        return await self.executor.cancel_order(order)

    async def modify(self, order: Order, **fields) -> Order:
        return await self.executor.modify_order(order, **fields)

    async def close_all(self, positions: List) -> int:
        closed = 0
        for p in positions:
            await self.executor.close_position(p)
            closed += 1
        return closed

    def get_order(self, client_order_id: str) -> Optional[Order]:
        return self._seen_client_ids.get(client_order_id)

    def all_orders(self) -> List[Order]:
        return list(self._all_orders)

    def open_orders(self) -> List[Order]:
        return list(self._open_orders.values())

    def has_signal(self, signal_id: str) -> bool:
        return any(o.signal_id == signal_id for o in self._all_orders)
