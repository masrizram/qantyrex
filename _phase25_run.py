"""Phase 25-27 real-data holding period analysis. DIAGNOSTIC ONLY."""
import ccxt, time, sys, dataclasses
import numpy as np
import trading_bot.config as C
from trading_bot.data.historical_ingestor import HistoricalIngestor, IngestConfig
from trading_bot.execution.signal_tracer import SignalTracer
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.backtest.holding_analysis import analyze_holding_period, holding_period_bottleneck_report

cfg = C.load_config()
cfg = dataclasses.replace(cfg, exchange="gate")
ex = ccxt.gate({"enableRateLimit": True})
since = int((time.time() - 365 * 24 * 3600) * 1000)

print("=== Fetch BTC/USDT 1h ===", flush=True)
icfg = IngestConfig(exchange="gate", symbol="BTC/USDT", timeframe="1h",
                    since_ms=since, limit_per_page=1000, max_pages=10,
                    cache_dir="./.data_cache", rate_limit_ms=200)
ing = HistoricalIngestor(ex, icfg)
df, rep = ing.ingest()
print("rows:", rep.candle_count, flush=True)

print("\n=== Acquire accepted signals ===", flush=True)
tracer = SignalTracer(cfg)
rm = RiskManager(cfg, equity=10_000)
report, accepted_signals = tracer.run_trace(df, rm)
features = tracer.engine.build_features(df, df)
print("accepted_signals:", len(accepted_signals), flush=True)

print("\n=== PHASE 25: Holding Period Analysis ===", flush=True)
analyses = analyze_holding_period(cfg, df, accepted_signals, features)
print(holding_period_bottleneck_report(analyses), flush=True)

print("\n=== PHASE 25 ROOT CAUSE ===", flush=True)
if analyses:
    a = analyses[0]
    print(f"Holding period: {a.holding_bars} bars (~{a.holding_bars/24:.1f} days)", flush=True)
    print(f"SL updates: {a.sl_updates} (BE + trailing stop activations)", flush=True)
    print(f"Exit reason: {a.exit_reason}", flush=True)
    if a.bar_states:
        dist = [b.dist_to_tp_pct for b in a.bar_states]
        print(f"TP distance: min={min(dist):.2f}% mean={np.mean(dist):.2f}%", flush=True)
        near_tp = sum(1 for d in dist if d < 1.0)
        print(f"Bars within 1% of TP: {near_tp}/{len(dist)}", flush=True)
        be_activated = a.sl_final >= a.entry_price if a.side == "BUY" else a.sl_final <= a.entry_price
        print(f"Break-even activated: {be_activated}", flush=True)
        print(f"SL initial: {a.sl_initial:.2f} -> SL final: {a.sl_final:.2f}", flush=True)
        print(f"Price went from {a.entry_price:.2f} to {a.exit_price:.2f}", flush=True)
        print(f"Profit: {a.pnl:.2f}", flush=True)
    print("\nROOT CAUSE: The TP is set at RR=2.0 (price risk * 2), which is a large distance", flush=True)
    print("on 1h BTC. The position rarely reaches TP before the trailing stop catches it.", flush=True)
    print("The trailing stop and break-even protect the position but extend holding time.", flush=True)
else:
    print("No trades to analyze — INSUFFICIENT_SAMPLE", flush=True)

print("\n=== FINAL: INSUFFICIENT_SAMPLE / NO_VERIFIED_EDGE / LIVE BLOCKED ===", flush=True)