"""Health monitor: exchange connectivity, data freshness, process state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import time


@dataclass
class HealthCheck:
    name: str
    healthy: bool
    detail: str = ""
    ts: int = field(default_factory=lambda: int(time.time() * 1000))


class HealthMonitor:
    def __init__(self) -> None:
        self._checks: Dict[str, HealthCheck] = {}

    def update(self, name: str, healthy: bool, detail: str = "") -> None:
        self._checks[name] = HealthCheck(name=name, healthy=healthy, detail=detail)

    def all_healthy(self) -> bool:
        return all(c.healthy for c in self._checks.values()) if self._checks else False

    def status(self) -> Dict[str, HealthCheck]:
        return dict(self._checks)

    def failing(self) -> List[str]:
        return [n for n, c in self._checks.items() if not c.healthy]

    def summary(self) -> str:
        if not self._checks:
            return "no checks registered"
        lines = []
        for n, c in self._checks.items():
            tag = "OK" if c.healthy else "FAIL"
            lines.append(f"[{tag}] {n}: {c.detail}")
        return "\n".join(lines)


# Standard health checks
def check_exchange_connectivity(exchange) -> tuple[bool, str]:
    try:
        exchange.fetch_status()  # ccxt method
        return True, "ok"
    except AttributeError:
        # fall back to a cheap endpoint
        try:
            exchange.fetch_ohlcv("BTC/USDT", "1m", limit=1)
            return True, "ok"
        except Exception as e:
            return False, str(e)
    except Exception as e:
        return False, str(e)


def check_data_freshness(last_candle_ts_ms: int, max_age_seconds: int = 600) -> tuple[bool, str]:
    age = (int(time.time() * 1000) - last_candle_ts_ms) / 1000
    if age > max_age_seconds:
        return False, f"stale ({age:.0f}s)"
    return True, f"fresh ({age:.0f}s)"
