"""Strategy registry: each strategy is versioned, named, and discoverable."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..core.exceptions import TradingError
from ..core.models import Signal


@dataclass
class StrategyMeta:
    name: str
    version: str
    description: str
    config: dict
    config_hash: str
    entry_fn: Callable[..., Optional[Signal]]  # noqa: F821


class StrategyRegistry:
    def __init__(self) -> None:
        self._strategies: Dict[str, StrategyMeta] = {}

    def register(self, meta: StrategyMeta) -> None:
        key = f"{meta.name}@{meta.version}"
        if key in self._strategies:
            raise TradingError(f"Strategy already registered: {key}")
        self._strategies[key] = meta

    def get(self, name: str, version: str) -> StrategyMeta:
        key = f"{name}@{version}"
        if key not in self._strategies:
            raise TradingError(f"Unknown strategy {key}")
        return self._strategies[key]

    def list(self) -> List[str]:
        return sorted(self._strategies.keys())

    def all(self) -> List[StrategyMeta]:
        return list(self._strategies.values())


_REGISTRY = StrategyRegistry()


def get_global_registry() -> StrategyRegistry:
    return _REGISTRY
