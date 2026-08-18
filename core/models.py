"""Core data models (pydantic v2) used across the platform."""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

try:
    from pydantic import BaseModel, Field, ConfigDict
except Exception:  # pragma: no cover - fallback minimal shim
    BaseModel = object  # type: ignore

    def Field(default=None, **kw):
        return default

    def ConfigDict(**kw):
        return {}


from .enums import (
    Side, OrderType, OrderStatus, ExitReason, TrendState, RegimeState,
    RegimeAction, SystemState, LiveReadiness,
)


def _utcnow_ms() -> int:
    return int(time.time() * 1000)


def _uuid() -> str:
    return uuid.uuid4().hex


class Candle(BaseModel if isinstance(BaseModel, type) else object):
    symbol: str
    timeframe: str
    timestamp: int  # ms epoch, open-time of the candle
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True

    if isinstance(BaseModel, type):
        model_config = ConfigDict(frozen=True)

    def __init__(self, **kwargs):
        if isinstance(BaseModel, type):
            super().__init__(**kwargs)
        else:  # shim
            for k, v in kwargs.items():
                setattr(self, k, v)


class Signal(BaseModel if isinstance(BaseModel, type) else object):
    signal_id: str = Field(default_factory=_uuid)
    strategy_version: str
    symbol: str
    side: Side
    entry: float
    stop_loss: float
    take_profit: float
    rr: float
    score: float
    trend: TrendState
    regime: RegimeState
    regime_action: RegimeAction
    rsi: float
    atr: float
    atr_percent: float
    adx: float
    ema_fast: float
    ema_slow: float
    spread_percent: float
    timestamp: int = Field(default_factory=_utcnow_ms)
    features: dict = Field(default_factory=dict)

    if isinstance(BaseModel, type):
        model_config = ConfigDict(arbitrary_types_allowed=True)


class Order(BaseModel if isinstance(BaseModel, type) else object):
    order_id: str = Field(default_factory=_uuid)
    client_order_id: str = Field(default_factory=_uuid)
    signal_id: Optional[str] = None
    trade_id: Optional[str] = None
    strategy_version: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    fee: float = 0.0
    slippage: float = 0.0
    created_at: int = Field(default_factory=_utcnow_ms)
    updated_at: int = Field(default_factory=_utcnow_ms)
    exchange_order_id: Optional[str] = None
    reject_reason: Optional[str] = None

    if isinstance(BaseModel, type):
        model_config = ConfigDict(arbitrary_types_allowed=True)


class Position(BaseModel if isinstance(BaseModel, type) else object):
    trade_id: str = Field(default_factory=_uuid)
    signal_id: Optional[str] = None
    signal_timestamp: int = 0
    strategy_version: str
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    initial_stop_loss: float
    stop_loss: float
    take_profit: float
    break_even: bool = False
    trailing_stop: Optional[float] = None
    opened_at: int = Field(default_factory=_utcnow_ms)
    closed_at: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    realized_pnl: float = 0.0
    fees: float = 0.0
    risk_percent: float = 0.0
    max_favorable: Optional[float] = None
    max_adverse: Optional[float] = None
    sl_audit: list = Field(default_factory=list)

    if isinstance(BaseModel, type):
        model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def is_open(self) -> bool:
        return self.closed_at is None


class TradeRecord(BaseModel if isinstance(BaseModel, type) else object):
    """Immutable historical trade journal entry."""
    trade_id: str
    signal_id: str
    strategy_version: str
    config_hash: str
    code_version: str
    signal_timestamp: int
    opened_at: int
    closed_at: int
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    initial_stop_loss: float
    final_stop_loss: float
    take_profit: float
    fees: float
    slippage: float
    pnl: float
    r_multiple: float
    regime: RegimeState
    signal_score: float
    risk_percent: float
    exit_reason: ExitReason

    if isinstance(BaseModel, type):
        model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


def config_hash(payload: dict) -> str:
    """Stable hash of a configuration dict for versioning."""
    blob = repr(sorted((k, v) for k, v in payload.items() if v is not None))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
