"""Startup self-check: verifies configuration, connectivity, and all subsystems
before transitioning to RUNNING. Fail-closed: any CRITICAL failure blocks trading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .config import Config
from .core.enums import SystemState
from .core.exceptions import ConfigError


@dataclass
class CheckResult:
    name: str
    critical: bool
    passed: bool
    detail: str = ""


@dataclass
class SelfCheckReport:
    results: List[CheckResult] = field(default_factory=list)

    @property
    def all_critical_passed(self) -> bool:
        return all(r.passed for r in self.results if r.critical)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = []
        for r in self.results:
            tag = "OK" if r.passed else ("FAIL*" if r.critical else "FAIL")
            lines.append(f"[{tag}] {r.name}: {r.detail}")
        return "\n".join(lines)


def run_self_checks(cfg: Config,
                    checks: List[Tuple[str, bool, Callable[[], Tuple[bool, str]]]]
                    ) -> SelfCheckReport:
    """`checks`: list of (name, critical, callable) where callable returns (ok, detail)."""
    rep = SelfCheckReport()
    for name, critical, fn in checks:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"exception: {e}"
        rep.results.append(CheckResult(name=name, critical=critical,
                                       passed=ok, detail=detail))
    return rep


# Standard check factories ---------------------------------------------------

def check_config(cfg: Config) -> Tuple[bool, str]:
    if cfg.max_risk_percent < cfg.initial_risk_percent:
        return False, "max_risk < initial_risk"
    if cfg.emergency_drawdown <= cfg.daily_max_drawdown:
        return False, "emergency_dd <= daily_dd"
    if cfg.min_rr < 1.0:
        return False, "min_rr < 1.0"
    if cfg.max_consecutive_losses < 1:
        return False, "max_consecutive_losses < 1"
    if cfg.trading_mode == Config.__annotations__["trading_mode"].__class__("LIVE") \
            and not cfg.live_trading_allowed:
        return False, "LIVE mode but live_trading_allowed=False"
    return True, "config ok"


def check_credentials(cfg: Config) -> Tuple[bool, str]:
    # In PAPER/BACKTEST, credentials are optional; in LIVE they're critical.
    if cfg.trading_mode.value == "LIVE":
        if not (cfg.api_key and cfg.api_secret):
            return False, "missing API credentials for LIVE"
    return True, "credentials ok"


def check_database(url: str) -> Tuple[bool, str]:
    try:
        from .storage.database import init_engine, create_all
        init_engine(url)
        create_all()
        return True, "db ok"
    except Exception as e:
        return False, f"db error: {e}"


def check_strategy_registry() -> Tuple[bool, str]:
    from .strategy.strategy_registry import get_global_registry
    reg = get_global_registry()
    if not reg.list():
        return False, "no strategy registered"
    return True, f"{len(reg.list())} strategy(ies) registered"


def check_telegram(cfg: Config) -> Tuple[bool, str]:
    # Telegram is optional; only fail if token set but whitelist empty
    if cfg.telegram_bot_token and not cfg.telegram_allowed_chat_ids:
        return False, "token set but whitelist empty"
    return True, "telegram ok" if not cfg.telegram_bot_token else "telegram armed"
