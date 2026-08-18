# Phase 26 Audit — Post-Phase-25 Quantitative Revalidation

**Date:** 2026-08-17 18:47:01 UTC
**Data source:** gate BTC/USDT 1h — 8761 candles (coverage 100.00%)
**Strategy:** `baseline@baseline_v1` — **UNMODIFIED**. Zero parameter changes.
**Tests:** 234 passing (all phases)
**LIVE TRADING:** **BLOCKED**

---

## 1. Regression Results

- **pytest:** 234 passed, 0 failed
- **compileall:** PASS — zero import errors, zero warnings

---

## 2. Data Provenance

- Row count: 8761
- Coverage ratio: 1.0000
- Dataset hash: c144a9c466b9e910
- Configuration hash: 1db4314a65acd561
- Exchange: gate
- Symbol: BTC/USDT
- Timeframe: 1h
- Since: 1755456359101
- Max pages: 10

---

## 3. Signal Parity (Phase 24 vs Phase 26)

| Metric | Phase 24 | Phase 26 | Delta |
|---|---|---|---|
| total_signals | 8760 | 8761 | 1 |
| accepted | 1 | 1 | 0 |
| ACCEPTED_NOT_OPENED | 0 | 0 | +0 |
| ACCEPTED_OPENED | 1 | 1 | +0 |
| BREAKER_STATE | 0 | 0 | +0 |
| DAILY_DD_LIMIT | 0 | 0 | +0 |
| DUPLICATE_SIGNAL | 0 | 0 | +0 |
| EXECUTION_GUARD | 0 | 0 | +0 |
| EXPOSURE_VIOLATION | 0 | 0 | +0 |
| LATENCY_OVERFLOW | 0 | 0 | +0 |
| LIQUIDITY_TOO_LOW | 0 | 0 | +0 |
| MAX_OPEN_POSITIONS | 442 | 442 | +0 |
| MOMENTUM_FAILED | 1487 | 1487 | +0 |
| NO_VALID_SL | 0 | 0 | +0 |
| PARTIAL_FILL | 0 | 0 | +0 |
| REGIME_NO_TRADE | 6386 | 6387 | +1 |
| REJECTION_SIMULATED | 0 | 0 | +0 |
| RISK_FACTOR_ZERO | 0 | 0 | +0 |
| SCORE_BELOW_MIN | 25 | 25 | +0 |
| SIZING_REJECTED | 0 | 0 | +0 |
| SL_ABOVE_ENTRY | 0 | 0 | +0 |
| SL_BELOW_ENTRY | 0 | 0 | +0 |
| SPREAD_TOO_HIGH | 0 | 0 | +0 |
| STATE_NOT_RUNNING | 0 | 0 | +0 |
| STRUCTURE_BLOCKS_TP | 25 | 25 | +0 |
| TREND_NEUTRAL | 384 | 384 | +0 |
| UNKNOWN | 0 | 0 | +0 |
| VOLATILITY_REJECTED | 10 | 10 | +0 |
| ZERO_RISK | 0 | 0 | +0 |

---

## 4. Trade Lifecycle Forensics

Lifecycle violations: 0
SL invariant violations: 0
Journal integrity issues: 0
Accounting violations: 0
Holding analysis violations: 0

All lifecycle invariants PASS.
All stop-loss invariants PASS.
All journal integrity checks PASS.
All accounting invariants PASS.
All holding analysis invariants PASS.

---

## 5. Trades

Trade count: 1
- trade_id=13de102989f84e11be3ace43b68afeda side=SELL entry=108445.58 exit=108467.27 initial_sl=109540.13 final_sl=108445.58 exit_reason=SL pnl=-9.00

---

## 6. Backtest Metrics

| Metric | Full | OOS |
|---|---|---|
| net_profit | -9.00 | 62.45 |
| profit_factor | 0.00 | inf |
| expectancy | -9.00 | 62.45 |
| expectancy_r | -0.22 | 1.57 |
| win_rate | 0.00 | 1.00 |
| max_drawdown | 0.0009 | 0.0000 |
| sharpe | 0.00 | 0.00 |
| sortino | 0.00 | 0.00 |
| calmar | -10000.00 | 0.00 |
| trade_count | 1 | 1 |
| total_fees | 8.1776 | 10.1729 |
| total_slippage | 0.8177 | 1.0248 |

---

## 7. Holding Analysis

- trade_id=d0591c5b8a6d45fc94f603f4e177ea4f holding_bars=3 exit_reason=SL sl_initial=109540.13 sl_final=108445.58 sl_updates=3 pnl=-9.00

---

## 8. Strategy Integrity

Integrity violations: 0
All parameters at default. No strategy tuning occurred.

---

## 9. Sample-Size Gates

- Full backtest: 1 trades (min: 100) -> INSUFFICIENT_SAMPLE
- OOS: 1 trades (min: 50) -> INSUFFICIENT_SAMPLE
- WF aggregate: 17 trades (min: 100) -> INSUFFICIENT_SAMPLE

---

## 10. Statistical Gates

- MC = NOT_EXECUTED_INSUFFICIENT_OOS_TRADES

---

## 11. LIVE Safety

- TRADING_MODE: PAPER
- LIVE_TRADING_ENABLED: False
- LIVE: BLOCKED

---

## 12. Final Classification

**Classification:** INSUFFICIENT_SAMPLE
**Reasons:** full_INSUFFICIENT_SAMPLE
**LIVE:** BLOCKED

---

## 13. Comparison (Phase 24 vs Phase 26)

- total_signals: P24=8760 P26=8761
- accepted_signals: P24=1 P26=1
- trades: P24=1 P26=1
- pnl: P24=393.59 P26=-8.995294236252454
- expectancy: P24=N/A P26=-8.995294236252454
- profit_factor: P24=N/A P26=0.0
- max_drawdown: P24=N/A P26=0.0008995294236252448
- sharpe: P24=N/A P26=0.0
- expectancy_r: P24=N/A P26=-0.21799140963546812
- holding_bars: P24=~1400 P26=3
- sl_updates: P24=N/A P26=3
- exit_reason: P24=timeout P26=SL
- classification: P24=INSUFFICIENT_SAMPLE P26=TBD
- live: P24=BLOCKED P26=TBD