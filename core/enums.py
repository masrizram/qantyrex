"""Enumerations used across the platform."""
from __future__ import annotations

from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ExitReason(str, Enum):
    TP = "TP"
    SL = "SL"
    BREAK_EVEN = "BREAK_EVEN"
    TRAILING = "TRAILING"
    MANUAL = "MANUAL"
    SIGNAL = "SIGNAL"
    RISK_LOCK = "RISK_LOCK"
    TIMEOUT = "TIMEOUT"


class TrendState(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class RegimeState(str, Enum):
    STRONG_TREND = "STRONG_TREND"
    WEAK_TREND = "WEAK_TREND"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class RegimeAction(str, Enum):
    TRADE = "TRADE"
    CONDITIONAL = "CONDITIONAL"
    NO_TRADE = "NO_TRADE"


class SystemState(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    RISK_LOCK = "RISK_LOCK"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


class LiveReadiness(str, Enum):
    REJECTED = "REJECTED"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    BACKTEST_VERIFIED = "BACKTEST_VERIFIED"
    OOS_VERIFIED = "OOS_VERIFIED"
    PAPER_TRADING = "PAPER_TRADING"
    MICRO_LIVE = "MICRO_LIVE"
    PRODUCTION_CANDIDATE = "PRODUCTION_CANDIDATE"
