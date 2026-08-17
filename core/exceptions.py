"""Domain-specific exception hierarchy. All failures fail-closed."""
from __future__ import annotations


class TradingError(Exception):
    """Base class for all trading-system errors."""


class ConfigError(TradingError):
    pass


class DataQualityError(TradingError):
    """Raised when market data fails validation and must not be used."""


class LookAheadBiasError(TradingError):
    """Raised when a feature/signal would leak future data."""


class RiskLimitBreached(TradingError):
    """Raised when a risk gate (DD, exposure, consecutive losses) is hit."""


class ExecutionError(TradingError):
    pass


class OrderRejected(ExecutionError):
    pass


class ReconciliationMismatch(TradingError):
    """Local state != exchange state and cannot be auto-resolved."""


class InvalidStateTransition(TradingError):
    pass


class StrategyDegraded(TradingError):
    pass


class AcceptanceGateFailed(TradingError):
    """Raised when a final acceptance gate (OOS, Monte Carlo, etc.) fails."""


class AuthorizationDenied(TradingError):
    pass
