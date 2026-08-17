"""Executor: exchange adapter abstraction (live + paper + backtest).

- LiveExecutor wraps a ccxt exchange instance.
- PaperExecutor simulates fills against the latest close (no real orders).
- BacktestExecutor is a no-op marker; backtests use the Simulator directly.

All executors share the same interface so the strategy / risk code is identical.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from ..core.enums import OrderStatus, OrderType, Side
from ..core.exceptions import ExecutionError, OrderRejected
from ..core.models import Order


class BaseExecutor:
    async def create_order(self, order: Order) -> Order: ...
    async def cancel_order(self, order: Order) -> Order: ...
    async def modify_order(self, order: Order, **fields) -> Order: ...
    async def close_position(self, position) -> None: ...
    async def get_positions(self) -> list: ...
    async def get_orders(self) -> list: ...
    async def get_balance(self) -> dict: ...
    async def get_order_status(self, order: Order) -> Order: ...


class PaperExecutor(BaseExecutor):
    """Paper trading: same logic as live but fills against the latest close."""

    def __init__(self, starting_balance: float = 10_000.0, fee_rate: float = 0.001,
                 slippage_bps: float = 2.0) -> None:
        self.balance = {"USDT": starting_balance}
        self.positions: dict[str, dict] = {}
        self.orders: list[Order] = []
        self.fee_rate = fee_rate
        self.slippage = slippage_bps / 10000.0
        self._last_prices: dict[str, float] = {}

    def set_market_price(self, symbol: str, price: float) -> None:
        self._last_prices[symbol] = price

    async def create_order(self, order: Order) -> Order:
        price = self._last_prices.get(order.symbol)
        if price is None:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "no_market_price"
            return order
        slip = self.slippage
        if order.side == Side.BUY:
            fill = price * (1 + slip)
        else:
            fill = price * (1 - slip)
        order.avg_fill_price = fill
        order.filled_quantity = order.quantity
        order.fee = order.quantity * fill * self.fee_rate
        order.status = OrderStatus.FILLED
        order.exchange_order_id = f"paper-{int(time.time()*1000)}"
        order.updated_at = int(time.time() * 1000)
        self.orders.append(order)
        # update balance/position
        cost = order.quantity * fill
        if order.side == Side.BUY:
            self.balance["USDT"] = self.balance.get("USDT", 0) - cost - order.fee
            self.positions[order.symbol] = {
                "side": order.side.value, "qty": order.quantity, "entry": fill}
        else:
            self.balance["USDT"] = self.balance.get("USDT", 0) + cost - order.fee
            self.positions.pop(order.symbol, None)
        return order

    async def cancel_order(self, order: Order) -> Order:
        order.status = OrderStatus.CANCELLED
        return order

    async def modify_order(self, order: Order, **fields) -> Order:
        for k, v in fields.items():
            setattr(order, k, v)
        return order

    async def close_position(self, position) -> None:
        sym = position.symbol
        price = self._last_prices.get(sym)
        if price is None:
            raise ExecutionError("no_market_price_to_close")
        slip = self.slippage
        fill = price * (1 - slip) if position.side == Side.BUY else price * (1 + slip)
        pnl = (fill - position.entry_price) * position.quantity if position.side == Side.BUY \
            else (position.entry_price - fill) * position.quantity
        self.balance["USDT"] = self.balance.get("USDT", 0) + pnl - position.fees
        self.positions.pop(sym, None)

    async def get_positions(self) -> list:
        return list(self.positions.values())

    async def get_orders(self) -> list:
        return list(self.orders)

    async def get_balance(self) -> dict:
        return dict(self.balance)

    async def get_order_status(self, order: Order) -> Order:
        return order


class LiveExecutor(BaseExecutor):
    """Wraps a ccxt exchange for live order submission.

    This is intentionally thin: the ExecutionGuard must approve before any
    order reaches here. ccxt's create_order returns a raw dict; we map it
    back onto our Order model.
    """

    def __init__(self, exchange, symbol: str) -> None:
        self.exchange = exchange
        self.symbol = symbol

    async def create_order(self, order: Order) -> Order:
        try:
            raw = self.exchange.create_order(
                symbol=order.symbol,
                type=order.order_type.value.lower(),
                side=order.side.value.lower(),
                amount=order.quantity,
                price=order.price,
                params={"clientOrderId": order.client_order_id},
            )
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(e)
            return order
        order.exchange_order_id = str(raw.get("id", ""))
        order.status = OrderStatus.SUBMITTED
        order.updated_at = int(time.time() * 1000)
        return order

    async def cancel_order(self, order: Order) -> Order:
        try:
            self.exchange.cancel_order(order.exchange_order_id, order.symbol)
            order.status = OrderStatus.CANCELLED
        except Exception as e:
            order.reject_reason = str(e)
        return order

    async def modify_order(self, order: Order, **fields) -> Order:
        # ccxt modify if supported
        try:
            self.exchange.edit_order(order.exchange_order_id, order.symbol,
                                     order.order_type.value.lower(),
                                     order.side.value.lower(),
                                     fields.get("quantity", order.quantity),
                                     fields.get("price", order.price))
            for k, v in fields.items():
                setattr(order, k, v)
        except Exception as e:
            order.reject_reason = str(e)
        return order

    async def close_position(self, position) -> None:
        # market close
        opp_side = Side.SELL if position.side == Side.BUY else Side.BUY
        o = Order(strategy_version=position.strategy_version, symbol=position.symbol,
                  side=opp_side, order_type=OrderType.MARKET, quantity=position.quantity)
        await self.create_order(o)

    async def get_positions(self) -> list:
        return self.exchange.fetch_positions([self.symbol])

    async def get_orders(self) -> list:
        return self.exchange.fetch_open_orders(self.symbol)

    async def get_balance(self) -> dict:
        return self.exchange.fetch_balance()

    async def get_order_status(self, order: Order) -> Order:
        raw = self.exchange.fetch_order(order.exchange_order_id, order.symbol)
        st = str(raw.get("status", "")).lower()
        mapping = {"open": OrderStatus.SUBMITTED, "closed": OrderStatus.FILLED,
                   "canceled": OrderStatus.CANCELLED, "rejected": OrderStatus.REJECTED,
                   "expired": OrderStatus.EXPIRED}
        order.status = mapping.get(st, OrderStatus.PENDING)
        if raw.get("filled"):
            order.filled_quantity = float(raw["filled"])
        if raw.get("average"):
            order.avg_fill_price = float(raw["average"])
        if raw.get("fee"):
            order.fee = float(raw["fee"].get("cost", 0))
        return order
