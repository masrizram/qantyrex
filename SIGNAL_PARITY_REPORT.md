# Signal Parity Report — Phase 24 vs Phase 26

**Date:** 2026-08-17 18:47:01 UTC
**Exchange:** gate
**Symbol:** BTC/USDT
**Timeframe:** 1h
**Dataset hash:** c144a9c466b9e910
**Configuration hash:** 1db4314a65acd561

## Summary

| Metric | Phase 24 | Phase 26 | Delta |
|---|---|---|---|
| Total signals | 8760 | 8761 | 1 |
| Accepted signals | 1 | 1 | 0 |

## Rejection Taxonomy

| Code | Phase 24 | Phase 26 | Delta | Status |
|---|---|---|---|---|
| ACCEPTED_NOT_OPENED | 0 | 0 | +0 | MATCH |
| ACCEPTED_OPENED | 1 | 1 | +0 | MATCH |
| BREAKER_STATE | 0 | 0 | +0 | MATCH |
| DAILY_DD_LIMIT | 0 | 0 | +0 | MATCH |
| DUPLICATE_SIGNAL | 0 | 0 | +0 | MATCH |
| EXECUTION_GUARD | 0 | 0 | +0 | MATCH |
| EXPOSURE_VIOLATION | 0 | 0 | +0 | MATCH |
| LATENCY_OVERFLOW | 0 | 0 | +0 | MATCH |
| LIQUIDITY_TOO_LOW | 0 | 0 | +0 | MATCH |
| MAX_OPEN_POSITIONS | 442 | 442 | +0 | MATCH |
| MOMENTUM_FAILED | 1487 | 1487 | +0 | MATCH |
| NO_VALID_SL | 0 | 0 | +0 | MATCH |
| PARTIAL_FILL | 0 | 0 | +0 | MATCH |
| REGIME_NO_TRADE | 6386 | 6387 | +1 | DIFFERENCE |
| REJECTION_SIMULATED | 0 | 0 | +0 | MATCH |
| RISK_FACTOR_ZERO | 0 | 0 | +0 | MATCH |
| SCORE_BELOW_MIN | 25 | 25 | +0 | MATCH |
| SIZING_REJECTED | 0 | 0 | +0 | MATCH |
| SL_ABOVE_ENTRY | 0 | 0 | +0 | MATCH |
| SL_BELOW_ENTRY | 0 | 0 | +0 | MATCH |
| SPREAD_TOO_HIGH | 0 | 0 | +0 | MATCH |
| STATE_NOT_RUNNING | 0 | 0 | +0 | MATCH |
| STRUCTURE_BLOCKS_TP | 25 | 25 | +0 | MATCH |
| TREND_NEUTRAL | 384 | 384 | +0 | MATCH |
| UNKNOWN | 0 | 0 | +0 | MATCH |
| VOLATILITY_REJECTED | 10 | 10 | +0 | MATCH |
| ZERO_RISK | 0 | 0 | +0 | MATCH |

## Unexplained Differences

- REGIME_NO_TRADE: P24=6386 P26=6387

## Invariant Verification
- total_signals == sum(all_terminal_outcomes): PASS
- Acceptance count: 1
- Rejection taxonomy completeness: PASS