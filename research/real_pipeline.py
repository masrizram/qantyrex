"""Phase 22 multi-asset real-data research pipeline.

Ingests real historical data via ccxt, runs the existing (UNMODIFIED) strategy
through the existing backtester/OOS/walk-forward/Monte-Carlo/sensitivity/stress
chain, and produces an honest per-asset + aggregate report.

Never modifies the strategy to improve results.
Never auto-enables LIVE. Maximum outcome = PAPER_TRADING_ELIGIBLE.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config import Config, load_config
from ..data.historical_ingestor import HistoricalIngestor, IngestConfig, dataset_hash
from ..data.quality_audit import audit as audit_quality, make_provenance, DataQualityReport, Provenance
from ..backtest.engine import Backtester, split_data
from ..backtest.metrics import metrics_to_dict
from ..backtest.walk_forward import walk_forward
from ..backtest.monte_carlo import monte_carlo
from ..backtest.sensitivity import analyze_sensitivity, overall_stability
from ..backtest.stress_test import run_stress, stress_summary, stress_passes
from .regime_coverage import classify_regime_coverage, per_regime_performance
from .sample_gates import (SampleGates, check_full_sample, check_oos_sample,
                             check_wf_sample, final_classification)


@dataclass
class AssetResult:
    symbol: str
    available: bool
    skip_reason: str = ""
    provenance: Optional[dict] = None
    quality: Optional[dict] = None
    backtest: Optional[dict] = None
    oos: Optional[dict] = None
    walk_forward: Optional[dict] = None
    monte_carlo: Optional[dict] = None
    sensitivity: Optional[dict] = None
    stress: Optional[dict] = None
    regimes: Optional[dict] = None
    classification: str = ""
    classification_reasons: List[str] = field(default_factory=list)
    sample_gates: Optional[dict] = None


def run_real_pipeline(
    cfg: Config,
    symbols: List[str],
    *,
    timeframe: str = "1h",
    limit_pages: int = 8,
    since_ms: Optional[int] = None,
    cache_dir: str = "./.data_cache",
    mc_iterations: int = 2000,
    wf_train: int = 1500,
    wf_oos: int = 300,
    wf_step: int = 300,
    gates: Optional[SampleGates] = None,
    exchange=None,
) -> List[AssetResult]:
    """Run the Phase 22 pipeline over `symbols`. Returns one AssetResult each.

    `exchange` is an already-initialized ccxt instance. If None, one is created
    via ccxt.<cfg.exchange>({enableRateLimit: True}).
    """
    gates = gates or SampleGates()
    results: List[AssetResult] = []

    if exchange is None:
        import ccxt
        ex_cls = getattr(ccxt, cfg.exchange, None)
        if ex_cls is None:
            raise ValueError(f"Unknown exchange {cfg.exchange!r}")
        exchange = ex_cls({"enableRateLimit": True})

    for sym in symbols:
        print(f"[phase22] {cfg.exchange} {sym} {timeframe} ...", flush=True)
        res = AssetResult(symbol=sym, available=True)
        try:
            icfg = IngestConfig(
                exchange=cfg.exchange, symbol=sym, timeframe=timeframe,
                since_ms=since_ms, limit_per_page=1000, max_pages=limit_pages,
                cache_dir=cache_dir,
            )
            ingestor = HistoricalIngestor(exchange, icfg)
            df, ing_rep = ingestor.ingest()
        except Exception as e:
            res.available = False
            res.skip_reason = f"ingest_failed: {type(e).__name__}: {str(e)[:200]}"
            results.append(res)
            print(f"[phase22]   SKIP {sym}: {res.skip_reason}", flush=True)
            continue

        if len(df) < (wf_train + wf_oos):
            res.available = False
            res.skip_reason = f"insufficient_data ({len(df)} < {wf_train + wf_oos})"
            results.append(res)
            print(f"[phase22]   SKIP {sym}: {res.skip_reason}", flush=True)
            continue

        # provenance + quality
        prov = make_provenance(df, cfg.exchange, sym, timeframe, ing_rep.configuration_hash)
        res.provenance = prov.to_dict()
        qrep = audit_quality(df, cfg.exchange, sym, timeframe, ing_rep.missing_ranges)
        res.quality = qrep.to_dict()
        print(f"[phase22]   quality={qrep.overall} rows={len(df)} gaps={len(qrep.missing_ranges)}", flush=True)
        data_ok = qrep.overall in ("PASS", "WARNING")

        bt = Backtester(cfg)

        # backtest (full)
        full = bt.run(df, split="FULL")
        fm = metrics_to_dict(full.result.metrics)
        res.backtest = fm

        # OOS
        tr, va, oos_df = split_data(df, (0.6, 0.2, 0.2))
        oos_res = bt.run(oos_df, split="OOS")
        om = metrics_to_dict(oos_res.result.metrics)
        res.oos = om

        # walk-forward
        def runner(window_df, split):
            return bt.run(window_df, split=split)
        wf = walk_forward(df, runner, train_size=wf_train, oos_size=wf_oos, step=wf_step)
        res.walk_forward = {
            "windows": len(wf.windows),
            "passing": wf.passing_windows,
            "failing": wf.failing_windows,
            "consistency": wf.consistency,
            "total_oos_trades": wf.total_oos_trades,
            "aggregate_oos_pf": wf.aggregate_oos_pf,
            "aggregate_oos_expectancy": wf.aggregate_oos_expectancy,
        }

        # Monte Carlo (only if sufficient OOS trades)
        if oos_res.result.metrics.trade_count >= gates.min_oos_trades:
            mc = monte_carlo(oos_res.result.trades, iterations=mc_iterations, seed=0)
            res.monte_carlo = {
                "executed": True, "iterations": mc.iterations,
                "median_return": mc.median_return, "p5_return": mc.p5_return,
                "p95_return": mc.p95_return, "median_dd": mc.median_dd,
                "p95_dd": mc.p95_dd, "probability_of_ruin": mc.probability_of_ruin,
            }
            mc_pass = mc.probability_of_ruin < 0.1 and mc.median_return > 0
        else:
            res.monte_carlo = {"executed": False, "reason": "INSUFFICIENT_OOS_TRADES"}
            mc_pass = False

        # sensitivity
        sens = analyze_sensitivity(
            {"ema_fast": cfg.ema_fast, "rsi_oversold": cfg.rsi_oversold, "min_rr": cfg.min_rr},
            {"ema_fast": [40, 50, 60], "rsi_oversold": [25, 30, 35], "min_rr": [1.8, 2.0, 2.2]},
            lambda overrides: _run_sens(cfg, tr, overrides),
        )
        res.sensitivity = {"stability": overall_stability(sens), "per_param": sens.stability}

        # stress (on OOS signals)
        features = bt.engine.build_features(oos_df, oos_df)
        stress_signals = []
        from ..core.models import Signal
        for i in range(len(features)):
            sr = bt.engine.evaluate(features, idx=i)
            if sr.signal is not None:
                stress_signals.append(sr.signal)
        stress_results = run_stress(oos_df, stress_signals, features)
        stress_ok = stress_passes(stress_results, min_pf=0.5, max_dd=0.6)
        res.stress = {"pass": stress_ok, "summary": stress_summary(stress_results)}

        # regime coverage + per-regime performance
        cov = classify_regime_coverage(df, sym)
        res.regimes = {"by_regime": cov.by_regime, "coverage_complete": cov.coverage_complete,
                       "per_regime_perf": per_regime_performance(oos_res.result.trades, df)}

        # sample gates + final classification
        gs = {
            "full_sample": check_full_sample(full.result.metrics, gates).status,
            "oos_sample": check_oos_sample(oos_res.result.metrics, gates).status,
            "wf_aggregate": check_wf_sample(wf.total_oos_trades, gates).status,
        }
        res.sample_gates = gs
        cls, reasons = final_classification(
            data_ok=data_ok,
            full_sample=check_full_sample(full.result.metrics, gates),
            oos_sample=check_oos_sample(oos_res.result.metrics, gates),
            wf_sample=check_wf_sample(wf.total_oos_trades, gates),
            oos_expectancy=om["expectancy"], oos_pf=om["profit_factor"],
            mc_executed=bool(res.monte_carlo.get("executed")), mc_pass=mc_pass,
            stress_pass=stress_ok,
        )
        res.classification = cls
        res.classification_reasons = reasons
        print(f"[phase22]   {sym} -> {cls}  ({', '.join(reasons)})", flush=True)
        results.append(res)

    return results


def _run_sens(cfg: Config, train_df: pd.DataFrame, overrides: dict) -> dict:
    import dataclasses as _dc
    fields = {}
    for k, v in overrides.items():
        if k in ("ema_fast",): fields[k] = int(v)
        elif k == "rsi_oversold": fields["rsi_oversold"] = float(v)
        elif k == "min_rr": fields["min_rr"] = float(v)
    c = _dc.replace(cfg, **fields) if fields else cfg
    btr = Backtester(c)
    r = btr.run(train_df, split="SENS")
    m = r.result.metrics
    return {"expectancy": m.expectancy, "profit_factor": m.profit_factor}


def aggregate_report(results: List[AssetResult]) -> dict:
    """Aggregate across assets — never cherry-pick. Report every symbol."""
    agg = {
        "total_symbols": len(results),
        "available": sum(1 for r in results if r.available),
        "skipped": [r.symbol for r in results if not r.available],
        "classifications": {},
        "regime_coverage": {},
    }
    for r in results:
        agg["classifications"][r.symbol] = r.classification
        if r.regimes:
            agg["regime_coverage"][r.symbol] = r.regimes.get("by_regime", {})
    # overall classification: best available, but never LIVE
    classes = [r.classification for r in results if r.available]
    if not classes:
        agg["overall"] = "INSUFFICIENT_DATA"
    elif all(c == "PAPER_TRADING_ELIGIBLE" for c in classes):
        agg["overall"] = "PAPER_TRADING_ELIGIBLE"
    elif any(c == "PAPER_TRADING_ELIGIBLE" for c in classes):
        agg["overall"] = "MIXED (some symbols paper-eligible)"
    else:
        agg["overall"] = "NO_VERIFIED_EDGE"
    return agg


def write_report(results: List[AssetResult], agg: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {"generated_at": int(time.time()*1000), "aggregate": agg,
               "assets": [asdict(r) for r in results]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
