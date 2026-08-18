"""Trade journal repository: immutable trade records + signal log."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import String, Float, Integer, BigInteger, Boolean, Index
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class TradeRecordORM(Base):
    __tablename__ = "trade_journal"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String, index=True, unique=True)
    signal_id: Mapped[str] = mapped_column(String, index=True)
    strategy_version: Mapped[str] = mapped_column(String, index=True)
    config_hash: Mapped[str] = mapped_column(String)
    code_version: Mapped[str] = mapped_column(String)
    timestamp: Mapped[int] = mapped_column(BigInteger, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    side: Mapped[str] = mapped_column(String)
    entry: Mapped[float] = mapped_column(Float)
    exit: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float)
    slippage: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float] = mapped_column(Float)
    r_multiple: Mapped[float] = mapped_column(Float)
    regime: Mapped[str] = mapped_column(String)
    signal_score: Mapped[float] = mapped_column(Float)
    risk_percent: Mapped[float] = mapped_column(Float)
    exit_reason: Mapped[str] = mapped_column(String)

    __table_args__ = (
        Index("ix_trade_symbol_ts", "symbol", "timestamp"),
    )


class SignalLogORM(Base):
    __tablename__ = "signal_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[str] = mapped_column(String, index=True, unique=True)
    strategy_version: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[int] = mapped_column(BigInteger, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    side: Mapped[str] = mapped_column(String)
    entry: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    rr: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    trend: Mapped[str] = mapped_column(String)
    regime: Mapped[str] = mapped_column(String)
    rsi: Mapped[float] = mapped_column(Float)
    atr: Mapped[float] = mapped_column(Float)
    adx: Mapped[float] = mapped_column(Float)
    spread_percent: Mapped[float] = mapped_column(Float)


from ..core.models import TradeRecord  # noqa: E402


def trade_record_to_orm(t: TradeRecord, config_hash: str, code_version: str) -> TradeRecordORM:
    return TradeRecordORM(
        trade_id=t.trade_id, signal_id=t.signal_id,
        strategy_version=t.strategy_version, config_hash=config_hash,
        code_version=code_version, timestamp=t.signal_timestamp,
        symbol=t.symbol, side=t.side.value, entry=t.entry_price, exit=t.exit_price,
        quantity=t.quantity, stop_loss=t.initial_stop_loss, take_profit=t.take_profit,
        fees=t.fees, slippage=t.slippage, pnl=t.pnl, r_multiple=t.r_multiple,
        regime=t.regime.value, signal_score=t.signal_score,
        risk_percent=t.risk_percent, exit_reason=t.exit_reason.value,
    )


class TradeJournalRepository:
    def __init__(self, session_factory) -> None:
        self.SessionLocal = session_factory

    def insert_trade(self, t: TradeRecordORM) -> None:
        with self.SessionLocal() as s:
            s.add(t)
            s.commit()

    def insert_many(self, trades: List[TradeRecordORM]) -> None:
        with self.SessionLocal() as s:
            for t in trades:
                s.add(t)
            s.commit()

    def all_trades(self) -> List[TradeRecordORM]:
        with self.SessionLocal() as s:
            return s.query(TradeRecordORM).order_by(TradeRecordORM.timestamp).all()

    def trades_for_symbol(self, symbol: str) -> List[TradeRecordORM]:
        with self.SessionLocal() as s:
            return s.query(TradeRecordORM).filter(
                TradeRecordORM.symbol == symbol
            ).order_by(TradeRecordORM.timestamp).all()

    def trade_count(self) -> int:
        with self.SessionLocal() as s:
            return s.query(TradeRecordORM).count()


__all__ = [
    "TradeRecordORM", "SignalLogORM", "TradeJournalRepository",
    "trade_record_to_orm",
]
