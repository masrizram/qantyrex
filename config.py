"""Central configuration loaded from environment / .env.

All values are validated and strongly typed. No credentials are ever logged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class TradingMode(str, Enum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


def _get_bool(key: str, default: str = "false") -> bool:
    return os.getenv(key, default).strip().lower() in {"1", "true", "yes", "on"}


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for {key}: {os.getenv(key)!r}") from exc


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid int for {key}: {os.getenv(key)!r}") from exc


def _get_list_int(key: str, default: str = "") -> List[int]:
    raw = os.getenv(key, default) or ""
    return [int(x.strip()) for x in raw.split(",") if x.strip().lstrip("-").isdigit()]


@dataclass(frozen=True)
class Config:
    # Trading mode
    trading_mode: TradingMode = field(default_factory=lambda: TradingMode(
        os.getenv("TRADING_MODE", "PAPER").upper()))

    # Exchange
    exchange: str = field(default_factory=lambda: os.getenv("EXCHANGE", "binance"))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("API_SECRET", ""))

    # Telegram
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_allowed_chat_ids: List[int] = field(
        default_factory=lambda: _get_list_int("TELEGRAM_ALLOWED_CHAT_IDS"))

    # Symbol / timeframes
    symbol: str = field(default_factory=lambda: os.getenv("SYMBOL", "BTC/USDT"))
    timeframe_trend: str = field(default_factory=lambda: os.getenv("TIMEFRAME_TREND", "1h"))
    timeframe_entry: str = field(default_factory=lambda: os.getenv("TIMEFRAME_ENTRY", "15m"))

    # Risk
    initial_risk_percent: float = field(default_factory=lambda: _get_float("INITIAL_RISK_PERCENT", 0.005))
    max_risk_percent: float = field(default_factory=lambda: _get_float("MAX_RISK_PERCENT", 0.01))
    daily_max_drawdown: float = field(default_factory=lambda: _get_float("DAILY_MAX_DRAWDOWN", 0.03))
    emergency_drawdown: float = field(default_factory=lambda: _get_float("EMERGENCY_DRAWDOWN", 0.05))
    max_consecutive_losses: int = field(default_factory=lambda: _get_int("MAX_CONSECUTIVE_LOSSES", 4))
    max_open_positions: int = field(default_factory=lambda: _get_int("MAX_OPEN_POSITIONS", 1))
    break_even_r: float = field(default_factory=lambda: _get_float("BREAK_EVEN_R", 1.0))
    min_rr: float = field(default_factory=lambda: _get_float("MIN_RR", 2.0))

    # Strategy
    ema_fast: int = field(default_factory=lambda: _get_int("EMA_FAST", 50))
    ema_slow: int = field(default_factory=lambda: _get_int("EMA_SLOW", 200))
    rsi_period: int = field(default_factory=lambda: _get_int("RSI_PERIOD", 14))
    atr_period: int = field(default_factory=lambda: _get_int("ATR_PERIOD", 14))
    adx_period: int = field(default_factory=lambda: _get_int("ADX_PERIOD", 14))
    atr_min_percent: float = field(default_factory=lambda: _get_float("ATR_MIN_PERCENT", 0.20))
    atr_max_percent: float = field(default_factory=lambda: _get_float("ATR_MAX_PERCENT", 3.00))
    max_spread_percent: float = field(default_factory=lambda: _get_float("MAX_SPREAD_PERCENT", 0.10))
    adx_min: float = field(default_factory=lambda: _get_float("ADX_MIN", 20.0))
    rsi_oversold: float = field(default_factory=lambda: _get_float("RSI_OVERSOLD", 30.0))
    rsi_overbought: float = field(default_factory=lambda: _get_float("RSI_OVERBOUGHT", 70.0))

    # Execution
    live_trading_enabled: bool = field(default_factory=lambda: _get_bool("LIVE_TRADING_ENABLED", "false"))
    fee_rate: float = field(default_factory=lambda: _get_float("FEE_RATE", 0.001))
    slippage_bps: float = field(default_factory=lambda: _get_float("SLIPPAGE_BPS", 2.0))

    # Persistence
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./trading_bot.db"))

    # Minimum signal score (0-100)
    min_signal_score: float = 75.0

    # Monte Carlo
    monte_carlo_iterations: int = 10000

    # Versioning (immutable per deployment)
    strategy_version: str = "baseline_v1"
    code_version: str = "0.1.0"

    # ----- Derived / safety -----
    @property
    def live_trading_allowed(self) -> bool:
        """True only when ALL live conditions are met (fail-closed)."""
        return (
            self.trading_mode == TradingMode.LIVE
            and self.live_trading_enabled is True
            and bool(self.api_key)
            and bool(self.api_secret)
        )

    def safe_dict(self) -> dict:
        """Return config dict with secrets redacted."""
        d = asdict(self)
        for k in ("api_key", "api_secret", "telegram_bot_token"):
            if d.get(k):
                d[k] = "***REDACTED***"
        return d


def load_config() -> Config:
    return Config()
