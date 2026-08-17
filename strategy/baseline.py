"""Baseline strategy: trend-following with momentum + structure confirmation.

This is the requested baseline. It is NOT assumed to be optimal; the research
engine (Phase 7) evaluates alternative hypotheses against it. The baseline is
registered under `baseline@baseline_v1`.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config import Config
from ..core.models import Signal
from .signal_engine import SignalEngine, SignalResult
from .strategy_registry import StrategyMeta, get_global_registry


def baseline_entry(
    engine: SignalEngine,
    features: pd.DataFrame,
    idx: int,
    spread_percent: float = 0.0,
    liquidity_ok: bool = True,
    latency_ms: Optional[float] = None,
) -> SignalResult:
    """Evaluate the baseline at closed bar `idx`."""
    return engine.evaluate(features, idx, spread_percent, liquidity_ok, latency_ms)


def register_baseline(cfg: Config, engine: SignalEngine) -> StrategyMeta:
    meta = StrategyMeta(
        name="baseline",
        version=cfg.strategy_version,
        description="EMA50/200 trend + RSI/MACD/ADX momentum + regime + structure",
        config=cfg.safe_dict(),
        config_hash=_hash_cfg(cfg),
        entry_fn=baseline_entry,
    )
    get_global_registry().register(meta)
    return meta


def _hash_cfg(cfg: Config) -> str:
    from ..core.models import config_hash
    return config_hash(cfg.safe_dict())
