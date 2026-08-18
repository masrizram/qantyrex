"""Phase 25 forensic trace: follow ONE signal through the entire lifecycle.
Prints every intermediate value. DIAGNOSTIC ONLY."""
import ccxt, time, sys, dataclasses, json
import numpy as np
import trading_bot.config as C
from trading_bot.data.historical_ingestor import HistoricalIngestor, IngestConfig
from trading_bot.execution.signal_tracer import SignalTracer
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.backtest.simulator import Simulator, SimulatorConfig
from trading_bot.core.enums import Side

cfg = C.load_config()
cfg = dataclasses.replace(cfg, exchange="gate")
ex = ccxt.gate({"enableRateLimit": True})
since = int((time.time() - 365 * 24 * 3600) * 1000)

print("=== FETCH ===", flush=True)
icfg = IngestConfig(exchange="gate", symbol="BTC/USDT", timeframe="1h",
                    since_ms=since, limit_per_page=1000, max_pages=10,
                    cache_dir="./.data_cache", rate_limit_ms=200)
ing = HistoricalIngestor(ex, icfg)
df, rep = ing.ingest()
print("rows:", rep.candle_count, flush=True)

print("\n=== TRACE: acquire accepted signals ===", flush=True)
tracer = SignalTracer(cfg)
rm = RiskManager(cfg, equity=10_000)
report, accepted_signals = tracer.run_trace(df, rm)
features = tracer.engine.build_features(df, df)
print("total_signals:", report.total_signals, "accepted:", len(accepted_signals), flush=True)

if not accepted_signals:
    print("NO ACCEPTED SIGNALS — INSUFFICIENT_SAMPLE", flush=True)
    sys.exit(0)

print("\n=== FORENSIC: signal object values ===", flush=True)
for s in accepted_signals[:3]:
    print("  signal_id:", s.signal_id[:12], flush=True)
    print("  side:", s.side.value, flush=True)
    print("  entry:", s.entry, flush=True)
    print("  stop_loss:", s.stop_loss, flush=True)
    print("  take_profit:", s.take_profit, flush=True)
    print("  score:", s.score, flush=True)
    print("  risk=|entry-sl|:", abs(s.entry - s.stop_loss), flush=True)
    print("  rr:", s.rr, flush=True)
    print("  features:", dict(s.features or {}), flush=True)
    print("  ---", flush=True)

print("\n=== FORENSIC: simulator dump ===", flush=True)
sim = Simulator(SimulatorConfig(
    fee_rate=cfg.fee_rate, slippage_bps=cfg.slippage_bps,
    break_even_r=cfg.break_even_r, max_open_positions=99,
    seed=0, initial_equity=999_000,
))
sim_result = sim.run(df, accepted_signals, features_for_atr=features)
trades = sim_result.trades
print("trades:", len(trades), flush=True)
print("rejected_signals:", sim_result.rejected_signals, flush=True)
print("signal_rejections:", sim_result.signal_rejections, flush=True)

if len(trades):
    for _, tr in trades.iterrows():
        print("\n  TRADE:", flush=True)
        print("  trade_id:", tr.get("trade_id", "")[:12], flush=True)
        print("  signal_id:", tr.get("signal_id", "")[:12], flush=True)
        tr_ts = tr["timestamp"]
        from datetime import datetime, timezone
        print("  timestamp:", tr_ts, "->", datetime.fromtimestamp(int(tr_ts)/1000, tz=timezone.utc).isoformat(), flush=True)
        print("  side:", tr["side"], flush=True)
        print("  entry:", tr["entry"], flush=True)
        print("  exit:", tr["exit"], flush=True)
        print("  stop_loss (from trade record):", tr["stop_loss"], flush=True)
        print("  take_profit:", tr["take_profit"], flush=True)
        print("  pnl:", tr["pnl"], flush=True)
        print("  exit_reason:", tr["exit_reason"], flush=True)
        print("  fees:", tr["fees"], flush=True)
        print("  quantity:", tr["quantity"], flush=True)

        # Find the original signal
        orig = None
        for s in accepted_signals:
            if s.signal_id == tr.get("signal_id"):
                orig = s
                break
        if orig:
            print("\n  ORIGINAL SIGNAL:", flush=True)
            print("  sig.entry:", orig.entry, flush=True)
            print("  sig.stop_loss:", orig.stop_loss, flush=True)
            print("  sig.take_profit:", orig.take_profit, flush=True)
            print("  sig.side:", orig.side.value, flush=True)
            print("  risk=|entry-sl|:", abs(orig.entry - orig.stop_loss), flush=True)
            print("  NOTE: trade_record.stop_loss == sig.stop_loss?", tr["stop_loss"] == orig.stop_loss, flush=True)
            print("  NOTE: trade_record.entry == sig.entry?", tr["entry"] == orig.entry, flush=True)
        else:
            print("  ORIGINAL SIGNAL NOT FOUND by signal_id", flush=True)

print("\n=== FINAL: FORENSIC COMPLETE ===", flush=True)