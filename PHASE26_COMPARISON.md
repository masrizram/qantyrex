# Phase 26 Comparison — Phase 24 vs Phase 26

**Date:** 2026-08-17 18:47:01 UTC

## Key Metrics

| Metric | Phase 24 | Phase 26 | Notes |
|---|---|---|---|
| total_signals | 8760 | 8761 | delta=+1 |
| accepted_signals | 1 | 1 | delta=+0 |
| trades | 1 | 1 | delta=+0 |
| pnl | 393.59 | -8.995294236252454 | delta=-402.5853 |
| expectancy | N/A | -8.995294236252454 |  |
| profit_factor | N/A | 0.0 |  |
| max_drawdown | N/A | 0.0008995294236252448 |  |
| sharpe | N/A | 0.0 |  |
| expectancy_r | N/A | -0.21799140963546812 |  |
| holding_bars | ~1400 | 3 |  |
| sl_updates | N/A | 3 |  |
| exit_reason | timeout | SL |  |
| classification | INSUFFICIENT_SAMPLE | TBD |  |
| live | BLOCKED | TBD |  |

## Material Differences

### Unexplained Differences
- REGIME_NO_TRADE: P24=6386 P26=6387

## Strategy Integrity
- Parameters changed: 0
- Strategy version: baseline_v1 (unchanged)
- No entry parameters changed
- No RSI thresholds changed
- No MACD thresholds changed
- No ADX thresholds changed
- No score threshold changed
- No regime thresholds changed
- No RR parameters changed
- No position-sizing optimization
- No max-open tuning

## Forensics
- Lifecycle violations: 0
- SL invariant violations: 0
- Journal integrity issues: 0
- Accounting violations: 0
- Holding analysis violations: 0

## Final Classification
- Classification: INSUFFICIENT_SAMPLE
- LIVE: BLOCKED