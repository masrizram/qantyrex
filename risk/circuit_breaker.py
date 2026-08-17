"""Circuit breakers: aggregate trigger sources -> RISK_LOCK or SHUTDOWN."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from ..core.enums import SystemState


@dataclass
class BreakerState:
    name: str
    tripped: bool = False
    reason: str = ""
    severity: str = "lock"  # "lock" | "shutdown"


class CircuitBreaker:
    def __init__(self) -> None:
        self._breakers: Dict[str, BreakerState] = {}
        self._checks: Dict[str, Callable[[], tuple[bool, str]]] = {}

    def register(self, name: str, check: Callable[[], tuple[bool, str]],
                 severity: str = "lock") -> None:
        self._breakers[name] = BreakerState(name=name, severity=severity)
        self._checks[name] = check

    def evaluate(self) -> List[BreakerState]:
        tripped = []
        for name, check in self._checks.items():
            ok, reason = check()
            st = self._breakers[name]
            st.tripped = not ok
            st.reason = reason
            if not ok:
                tripped.append(st)
        return tripped

    def desired_state(self) -> SystemState:
        tripped = self.evaluate()
        if any(t.severity == "shutdown" for t in tripped):
            return SystemState.SHUTDOWN
        if tripped:
            return SystemState.RISK_LOCK
        return SystemState.RUNNING

    def reset(self, name: str | None = None) -> None:
        if name is None:
            for st in self._breakers.values():
                st.tripped = False; st.reason = ""
        elif name in self._breakers:
            self._breakers[name].tripped = False
            self._breakers[name].reason = ""

    def status(self) -> Dict[str, BreakerState]:
        return dict(self._breakers)
