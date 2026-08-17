"""Main entrypoint for the autonomous quant trading platform.

Usage:
  python -m trading_bot.main research            # run full validation pipeline
  python -m trading_bot.main run                  # run the live/paper loop
  python -m trading_bot.main selfcheck            # run startup self-checks

The system starts in PAPER mode by default and fails closed if live-trading
acceptance gates are not met. It NEVER fabricates profitability.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from .config import Config, load_config
from .core.enums import SystemState
from .core.exceptions import RiskLimitBreached
from .core.state_machine import StateMachine
from .data.market_data import make_synthetic_dataframe
from .pipeline import run_pipeline
from .startup import (
    run_self_checks, check_config, check_credentials, check_database,
    check_strategy_registry, check_telegram, SelfCheckReport,
)
from .strategy.baseline import register_baseline
from .strategy.signal_engine import SignalEngine

log = logging.getLogger("trading_bot")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def build_self_checks(cfg: Config):
    return [
        ("config", True, lambda: check_config(cfg)),
        ("credentials", True, lambda: check_credentials(cfg)),
        ("database", False, lambda: check_database(cfg.database_url)),
        ("strategy_registry", True, check_strategy_registry),
        ("telegram", False, lambda: check_telegram(cfg)),
    ]


def cmd_selfcheck(cfg: Config) -> int:
    # ensure baseline is registered
    engine = SignalEngine(cfg)
    try:
        register_baseline(cfg, engine)
    except Exception:
        pass  # already registered
    rep = run_self_checks(cfg, build_self_checks(cfg))
    print(rep.summary())
    return 0 if rep.all_critical_passed else 1


def cmd_research(cfg: Config) -> int:
    log.info("Running full research/validation pipeline on synthetic data...")
    # Use synthetic data because no live exchange data is configured by default.
    # This is HONEST: the pipeline exercises every subsystem, but any edge
    # found on synthetic GBM data has NO bearing on real-market edge.
    df = make_synthetic_dataframe(n=800, tf="1h", seed=2024)
    result = run_pipeline(df, cfg, mc_iterations=2000,
                          wf_train=400, wf_oos=100, wf_step=100)
    print(result.report_text)
    print("\n" + "=" * 50)
    from .backtest.report import scorecard as render_scorecard
    print(render_scorecard(result.scorecard))
    print("\nFINAL DECISION: " + result.readiness.value)
    if not result.statistical_edge:
        print("STRATEGY REJECTED  ::  NO VERIFIED EDGE  ::  LIVE TRADING BLOCKED")
    return 0


async def _run_loop(cfg: Config) -> int:
    sm = StateMachine()
    sm.transition(SystemState.READY)
    rep = run_self_checks(cfg, build_self_checks(cfg))
    print(rep.summary())
    if not rep.all_critical_passed:
        print("CRITICAL SELF-CHECK FAILED -> DO NOT TRADE")
        return 1
    sm.transition(SystemState.RUNNING)
    log.info("Bot RUNNING in %s mode", cfg.trading_mode.value)
    # In a real deployment this would loop: fetch data -> features -> signals
    # -> risk -> execution -> reconciliation -> monitoring. For the CLI demo we
    # simply demonstrate the lifecycle and exit cleanly.
    log.info("Lifecycle OK. (Production loop omitted in CLI demo.)")
    sm.transition(SystemState.SHUTDOWN)
    return 0


def cmd_run(cfg: Config) -> int:
    engine = SignalEngine(cfg)
    try:
        register_baseline(cfg, engine)
    except Exception:
        pass
    return asyncio.run(_run_loop(cfg))


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="trading_bot")
    parser.add_argument("command", choices=["research", "run", "selfcheck"],
                        help="What to do")
    args = parser.parse_args(argv)
    cfg = load_config()
    if args.command == "research":
        return cmd_research(cfg)
    if args.command == "selfcheck":
        return cmd_selfcheck(cfg)
    if args.command == "run":
        return cmd_run(cfg)
    return 2


if __name__ == "__main__":
    sys.exit(main())
