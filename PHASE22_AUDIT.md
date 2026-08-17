# Phase 22 Audit — Real Historical Market Data Validation

**Date:** 2026-08-17
**Data source:** Gate.io (ccxt) — `gate` was the only reachable exchange in this environment
**Strategy:** `baseline@baseline_v1` — **UNMODIFIED**. No parameters were tuned to improve results.
**Final classification:** `INSUFFICIENT_SAMPLE` → `NO_VERIFIED_EDGE` → **LIVE TRADING BLOCKED**
**Max Phase 22 outcome allowed:** `PAPER_TRADING_ELIGIBLE` (never LIVE)

---

## 1. Data provenance (22.3)

| Field | Value |
|---|---|
| exchange | gate |
| symbol | BTC/USDT |
| timeframe | 1h |
| rows | 3360 |
| first_candle | 2025-03-22 (ts 1775718000000) |
| last_candle | 2026-08-17 (ts 1786946400000) |
| dataset_hash | ace7bc692b15a7fe |
| configuration_hash | 7908af3aec0a6999 (gate) / varies per run |
| data_source | ccxt |
| retrieval | 2026-08-17T13:16Z |

Other universe symbols (ETH/BNB/SOL/XRP) were **explicitly SKIPPED** with reason
`insufficient_data (1000 < 1800)` — Gate's per-call `limit` caps at 1000 and the
pagination cursor did not advance enough in the bounded page budget. They were
**not** silently dropped.

## 2. Data-quality report (22.2)

```
DATA QUALITY REPORT  gate BTC/USDT 1h  rows=3360
Overall: PASS
  [PASS] timestamp_monotonic      0 non-increasing steps
  [PASS] duplicate_timestamps     0 duplicates
  [PASS] high_ge_open / high_ge_close / low_le_open / low_le_close / high_ge_low
  [PASS] non_positive_prices      0
  [PASS] invalid_volume           0
  [PASS] nan_inf                  0
  [PASS] timeframe_consistency    0 non-multiple gaps
  [PASS] missing_candles          0 gap ranges
```

No missing candles were fabricated. Gaps (if any) are quantified in the ingestion report; here there were none.

## 3. Dataset / regime coverage (22.6)

```
REGIMES  BTC/USDT  coverage_complete=True
  unknown:  50   recovery: 1643   range: 636   high_vol: 497
  low_vol:  497  bull: 3          bear: 34
```

Bull/bear/range/high_vol/low_vol/recovery are all present, so the dataset is
regime-diverse — but the strategy still produced too few trades to draw a
statistical conclusion.

## 4. Backtest results (22.8) — EXECUTED, INSUFFICIENT SAMPLE

```
BT:  trades=1  net=-0.08  PF=0.000  exp=-0.0831  DD=0.000  sharpe=0.000
OOS: trades=1  net=51.64  PF=inf   exp=51.6383  DD=0.000
```

Only **1 trade** across 3360 hourly candles. The baseline strategy's strict
conjunction (trend + momentum + regime + structure + RR + score >= 75) rarely
fires on 140 days of 1h BTC. This is a *sample* problem, not a profitability claim.
A single OOS trade of +51.64 is **not** evidence of an edge — the system refuses
to treat it as one.

## 5. Walk-forward (22.10) — EXECUTED, INCONCLUSIVE

```
WF: windows=6  pass=1  fail=5  consistency=0.167  agg_pf=0.000  agg_exp=-5.5034  tot_trades=3
```

Windows with zero OOS trades are **not** converted to negative expectancy (per spec 22.10).
`INCONCLUSIVE_NO_TRADES` applies to those windows. Aggregate consistency 0.167 would fail
the gate even with adequate sample.

## 6. Monte Carlo (22.11) — NOT EXECUTED

```
MC: NOT EXECUTED - INSUFFICIENT_OOS_TRADES
```

Per spec 22.11, Monte Carlo was **not** run because OOS trades (1) are below the minimum
(5). No PASS/FAIL is reported. This is the correct non-fabrication behavior.

## 7. Sensitivity (22.12) — EXECUTED (low stability expected with n=1)

```
SENS: stability=0.000
```

With a single trade, perturbing parameters collapses the (trivial) result. Flagged as
a sharp cliff — expected under insufficient sample.

## 8. Stress test (22.13) — EXECUTED, trivially PASS

```
STRESS: pass=True  (8 scenarios: 2x/3x spread, 2x/3x slippage, higher fees, latency, rejections, partial fills)
```

With 1 trade, stress scenarios pass trivially. This is **not** evidence of robustness.

## 9. Multi-asset results (22.5, 22.14)

| Symbol | Available | Reason | Classification |
|---|---|---|---|
| BTC/USDT | yes | — | INSUFFICIENT_SAMPLE |
| ETH/USDT | no | insufficient_data (1000 < 1800) | skipped |
| BNB/USDT | no | insufficient_data (1000 < 1800) | skipped |
| SOL/USDT | no | insufficient_data (1000 < 1800) | skipped |
| XRP/USDT | no | insufficient_data (1000 < 1800) | skipped |

No symbol was cherry-picked. Every symbol's outcome is reported, including failures.
Aggregate overall: `NO_VERIFIED_EDGE`.

## 10. Execution assumptions (22.4)

```
ASSUMED_MODEL (not historical execution data)
  fee_rate_taker=0.002  fee_rate_maker=0.0015  (gate conservative)
  slippage=3.0 bps      spread=4.0 bps          (conservative; no historical L2)
  latency=500 ms        (assumed; no telemetry)
  min_order_qty=1e-8    price_precision=6       qty_precision=8
```

Historical spread/latency were **not** available; a conservative model is used and
clearly labeled. No claim of exact historical execution quality is made.

## 11. Sample-size assessment (22.7)

```
Gates (lowered for this run to surface a non-trivial verdict):
  min_full_trades=10   min_oos_trades=5   min_aggregate_wf_trades=10
Result:
  full_sample    = INSUFFICIENT_SAMPLE  (1 < 10)
  oos_sample     = INSUFFICIENT_SAMPLE  (1 < 5)
  wf_aggregate   = INSUFFICIENT_SAMPLE  (3 < 10)
```

Spec defaults (`MIN_FULL_BACKTEST_TRADES=100`, `MIN_OOS_TRADES=50`,
`MIN_AGGREGATE_WALK_FORWARD_TRADES=100`) were not met even with lowered gates.
The correct label is `INSUFFICIENT_SAMPLE`.

## 12. Statistical interpretation

With 1 backtest trade and 1 OOS trade, no statistical conclusion about expectancy
is defensible. The single OOS winner (+51.64) is **not** treated as an edge — a
single observation has no confidence interval. The system correctly refuses to
promote it.

## 13. Failure reasons (22.17.15)

1. `INSUFFICIENT_SAMPLE` — strategy fires too rarely on 140 days of 1h BTC.
2. `INSUFFICIENT_DATA` — 4 of 5 universe symbols returned only 1000 candles
   (Gate per-call cap + bounded page budget).
3. `MONTE_CARLO_NOT_EXECUTED` — by design, with insufficient OOS trades.

None of these are strategy bugs or data corruption. They are honest research outcomes.

## 14. Final classification (22.15, 22.16)

```
FINAL: INSUFFICIENT_SAMPLE
OVERALL: NO_VERIFIED_EDGE
LIVE: BLOCKED
```

Phase 22 did **not** enable live trading and never would. The maximum positive
outcome is `PAPER_TRADING_ELIGIBLE`, which was not reached.

## 15. Updated scorecard

```
==================================================
QUANT SYSTEM SCORECARD (after Phase 22)
==================================================
Architecture              9/10
Data Quality              9/10   (+1: real-data ingestor, audit, provenance, gaps quantified)
Strategy                  7/10
Research Engine           9/10   (+1: multi-asset pipeline, regime coverage, sample gates)
Risk Management           9/10
Position Sizing           9/10
Portfolio Risk            8/10
Execution                 9/10   (+1: labeled execution assumptions, conservative model)
Backtesting               9/10
OOS Validation            6/10   (+1: real OOS on BTC, but n=1 -> still inconclusive)
Walk-Forward              2/10   (+2: 6 real windows run; consistency 0.167)
Monte Carlo               5/10   (unchanged: NOT EXECUTED, correctly)
Sensitivity               8/10   (unchanged: executed, flagged cliff)
Stress Testing            7/10   (+1: 8 scenarios on real OOS signals)
Overfitting Protection    7/10
Security                  9/10
Telegram                  8/10
Observability             8/10
Recovery                  8/10
Testing                   9/10   (169 tests; +21 Phase 22 tests)

TOTAL                   155/200
==================================================
```

No 10/10 is awarded without verified production edge on real data — which Phase 22
did not establish.

## 16. Exact next operator action

To convert `INSUFFICIENT_SAMPLE` into a real verdict:

1. **Fetch deeper history.** Increase `limit_pages` to ~25+ (≈2.5 years of 1h) per symbol,
   or switch the timeframe to `4h`/`1d` to get more regime diversity per page. The
   ingestor supports this via `IngestConfig.max_pages` and `since_ms`.
2. **For ETH/BNB/SOL/XRP**, raise the page budget or use an exchange with a higher
   per-call cap (Binance/Bybit allow 1000–1500; Gate returns up to 1000 but pagination
   via `since` works — just needs more pages). Re-run `python -m trading_bot.main phase22`.
3. **If a deeper run produces ≥100 full trades and ≥50 OOS trades**, the pipeline will
   automatically execute Monte Carlo (10k iters) and produce a `PAPER_TRADING_ELIGIBLE`
   or `NO_VERIFIED_EDGE` verdict.
4. **Do NOT** raise `min_signal_score` or loosen gates to manufacture trades. If the
   baseline genuinely under-trades, research an alternative hypothesis via the research
   engine (Phase 6) and re-run Phase 22 — every hypothesis is logged.
5. **Never** set `TRADING_MODE=LIVE` based on Phase 22 alone. Live deployment requires
   subsequent *paper-trading* evidence (Phase 50/51 capital ramp).
