"""Sample-size gates and final classification for Phase 22.

Insufficient sample -> INSUFFICIENT_SAMPLE; never a misleading conclusion.
Zero-trade windows -> INCONCLUSIVE_NO_TRADES (NOT a negative expectancy).
The maximum positive outcome of Phase 22 is PAPER_TRADING_ELIGIBLE — never LIVE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..backtest.metrics import Metrics


@dataclass
class SampleGates:
    min_full_trades: int = 100
    min_oos_trades: int = 50
    min_aggregate_wf_trades: int = 100


@dataclass
class GateResult:
    name: str
    status: str  # PASS / FAIL / INSUFFICIENT_SAMPLE / INCONCLUSIVE / NOT_EXECUTED
    detail: str = ""


def check_full_sample(metrics: Metrics, gates: SampleGates) -> GateResult:
    n = metrics.trade_count
    if n < gates.min_full_trades:
        return GateResult("full_sample", "INSUFFICIENT_SAMPLE", f"{n} < {gates.min_full_trades}")
    return GateResult("full_sample", "PASS", f"{n} >= {gates.min_full_trades}")


def check_oos_sample(metrics: Metrics, gates: SampleGates) -> GateResult:
    n = metrics.trade_count
    if n == 0:
        return GateResult("oos_sample", "INCONCLUSIVE_NO_TRADES", "0 OOS trades")
    if n < gates.min_oos_trades:
        return GateResult("oos_sample", "INSUFFICIENT_SAMPLE", f"{n} < {gates.min_oos_trades}")
    return GateResult("oos_sample", "PASS", f"{n} >= {gates.min_oos_trades}")


def check_wf_sample(total_oos_trades: int, gates: SampleGates) -> GateResult:
    if total_oos_trades < gates.min_aggregate_wf_trades:
        return GateResult("wf_aggregate_sample", "INSUFFICIENT_SAMPLE",
                          f"{total_oos_trades} < {gates.min_aggregate_wf_trades}")
    return GateResult("wf_aggregate_sample", "PASS", "")


# Final classification labels (Phase 22.15)
REJECTED = "REJECTED"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
NO_VERIFIED_EDGE = "NO_VERIFIED_EDGE"
OOS_VERIFIED = "OOS_VERIFIED"
PAPER_TRADING_ELIGIBLE = "PAPER_TRADING_ELIGIBLE"


def final_classification(
    *,
    data_ok: bool,
    full_sample: GateResult,
    oos_sample: GateResult,
    wf_sample: GateResult,
    oos_expectancy: float,
    oos_pf: float,
    mc_executed: bool,
    mc_pass: bool,
    stress_pass: bool,
) -> tuple[str, List[str]]:
    """Return (classification, reasons). Never returns LIVE-related labels."""
    reasons: List[str] = []
    if not data_ok:
        reasons.append("data_quality_failed")
        return INSUFFICIENT_DATA, reasons
    if full_sample.status != "PASS":
        reasons.append(f"full_{full_sample.status}")
        return INSUFFICIENT_SAMPLE, reasons
    if oos_sample.status == "INCONCLUSIVE_NO_TRADES":
        reasons.append("oos_no_trades")
        return NO_VERIFIED_EDGE, reasons
    if oos_sample.status != "PASS":
        reasons.append(f"oos_{oos_sample.status}")
        return INSUFFICIENT_SAMPLE, reasons
    if wf_sample.status != "PASS":
        reasons.append(f"wf_{wf_sample.status}")
        return INSUFFICIENT_SAMPLE, reasons
    if oos_expectancy <= 0 or oos_pf <= 1.0:
        reasons.append(f"oos_edge_absent e={oos_expectancy:.4f} pf={oos_pf:.3f}")
        return NO_VERIFIED_EDGE, reasons
    if not mc_executed:
        reasons.append("monte_carlo_not_executed")
        return NO_VERIFIED_EDGE, reasons
    if not mc_pass:
        reasons.append("monte_carlo_failed")
        return REJECTED, reasons
    if not stress_pass:
        reasons.append("stress_failed")
        return REJECTED, reasons
    # all gates passed -> paper eligible (NEVER auto-LIVE)
    reasons.append("all_gates_passed")
    return PAPER_TRADING_ELIGIBLE, reasons
