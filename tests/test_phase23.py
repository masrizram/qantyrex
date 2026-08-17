"""Phase 23 regression tests: timestamp audit, pagination cursor, funnel,
rejection reasons, condition counts, diagnostic-only classification, counterfactual."""
import numpy as np
import pandas as pd
import pytest

from trading_bot.config import Config
from trading_bot.data.timestamp_audit import ts_to_utc_iso, ts_to_utc_date, audit_timestamps
from trading_bot.data.historical_ingestor import HistoricalIngestor, IngestConfig
from trading_bot.data.market_data import make_synthetic_dataframe
from trading_bot.strategy.diagnostic import (
    instrument_signals, condition_correlation_matrix, bottleneck_analysis,
    rejection_reason_distribution, counterfactual_progressive,
    per_regime_signal_frequency, multi_timeframe_diagnostic,
)
from trading_bot.backtest.simulator import Simulator


# ---- 23.1 Timestamp audit ----

def test_ts_to_utc_iso_known():
    assert ts_to_utc_iso(0).startswith("1970-01-01T00:00:00")
    assert ts_to_utc_date(0) == "1970-01-01"


def test_ts_to_utc_phase22_first_candle():
    assert ts_to_utc_date(1775718000000) == "2026-04-09"
    assert ts_to_utc_date(1786946400000) == "2026-08-17"


def test_audit_timestamps_passes_clean():
    df = make_synthetic_dataframe(n=300, tf="1h", seed=1)
    rep = audit_timestamps(df, timeframe="1h")
    assert rep.overall == "PASS"
    assert rep.duration_days > 0


def test_audit_timestamps_fails_on_claimed_date_mismatch():
    df = make_synthetic_dataframe(n=300, tf="1h", seed=2)
    rep = audit_timestamps(df, claimed_first_date="1999-01-01", timeframe="1h")
    assert rep.overall == "FAIL"
    assert any(c.name == "claimed_first_date_matches" and c.status == "FAIL" for c in rep.checks)


def test_audit_timestamps_fails_on_non_monotonic():
    df = make_synthetic_dataframe(n=100, tf="1h", seed=3).reset_index(drop=True).copy()
    t0 = df["timestamp"].iloc[0]
    df.loc[0, "timestamp"] = df.loc[50, "timestamp"]
    df.loc[50, "timestamp"] = t0
    rep = audit_timestamps(df, timeframe="1h")
    assert rep.overall == "FAIL"


def test_audit_timestamps_empty_fail():
    rep = audit_timestamps(pd.DataFrame(columns=["timestamp"]), timeframe="1h")
    assert rep.overall == "FAIL"


def test_phase22_provenance_discrepancy_regression():
    """Phase 22 audit text said first_candle=2025-03-22 but ts=1775718000000
    maps to 2026-04-09. This pins the correct date so the error is not reintroduced."""
    assert ts_to_utc_date(1775718000000) == "2026-04-09"
    assert ts_to_utc_date(1786946400000) == "2026-08-17"
    span_days = (1786946400000 - 1775718000000) / (1000 * 3600 * 24)
    assert 125 < span_days < 135


# ---- 23.3 Pagination cursor advancement ----

class _FakeExchange:
    def __init__(self, total=3000, tf_ms=3_600_000):
        self.total = total; self.tf = tf_ms; self.calls = 0
    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        self.calls += 1
        start = 0 if since is None else since
        page = []
        for i in range(limit):
            ts = start + i * self.tf
            if ts > self.total * self.tf:
                break
            page.append([ts, 100.0, 101.0, 99.0, 100.5, 1.0])
        return page


def test_pagination_records_pages_and_cursor_advances(tmp_path):
    ex = _FakeExchange(total=2500)
    cfg = IngestConfig(exchange="fake", symbol="BTC/USDT", timeframe="1h",
                       since_ms=0, limit_per_page=1000, max_pages=5,
                       cache_dir=str(tmp_path), rate_limit_ms=0)
    ing = HistoricalIngestor(ex, cfg)
    df, rep = ing.ingest()
    assert len(rep.pages) >= 2
    for p in rep.pages:
        if p.row_count > 0:
            assert p.cursor_advanced is True
            assert p.next_since > p.request_since
    assert rep.pagination_failures == 0


def test_pagination_detects_non_advancing_cursor(tmp_path):
    class _StuckExchange:
        calls = 0
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
            self.calls += 1
            return [[0, 100, 101, 99, 100.5, 1.0]] * 5
    ex = _StuckExchange()
    cfg = IngestConfig(exchange="stuck", symbol="BTC/USDT", timeframe="1h",
                       since_ms=0, limit_per_page=5, max_pages=3,
                       cache_dir=str(tmp_path), rate_limit_ms=0, max_retries=1)
    ing = HistoricalIngestor(ex, cfg)
    df, rep = ing.ingest()
    assert rep.pagination_failures >= 1


def test_pagination_dedup(tmp_path):
    ex = _FakeExchange(total=500)
    cfg = IngestConfig(exchange="fake", symbol="BTC/USDT", timeframe="1h",
                       since_ms=0, limit_per_page=1000, max_pages=2,
                       cache_dir=str(tmp_path), rate_limit_ms=0)
    ing = HistoricalIngestor(ex, cfg)
    df, rep = ing.ingest()
    assert df["timestamp"].duplicated().sum() == 0


def test_coverage_ratio_computed(tmp_path):
    ex = _FakeExchange(total=2000)
    cfg = IngestConfig(exchange="fake", symbol="BTC/USDT", timeframe="1h",
                       since_ms=0, limit_per_page=1000, max_pages=3,
                       cache_dir=str(tmp_path), rate_limit_ms=0)
    ing = HistoricalIngestor(ex, cfg)
    df, rep = ing.ingest()
    assert rep.expected_candles > 0
    assert 0.0 <= rep.coverage_ratio <= 1.0 + 1e-6


# ---- 23.4 / 23.6 Funnel + rejection reasons ----

def test_funnel_accounts_all_candles():
    cfg = Config()
    df = make_synthetic_dataframe(n=800, tf="1h", seed=5)
    funnel, mat = instrument_signals(cfg, df)
    assert funnel.total_candles == 800
    assert len(funnel.rows) == 6
    rates = [r.cumulative_rate for r in funnel.rows]
    for i in range(1, len(rates)):
        assert rates[i] <= rates[i-1] + 1e-9
    assert funnel.final_signals <= funnel.rows[-1].pass_count


def test_funnel_rejection_reasons_explicit():
    cfg = Config()
    df = make_synthetic_dataframe(n=800, tf="1h", seed=6)
    funnel, _ = instrument_signals(cfg, df)
    dist = rejection_reason_distribution(funnel)
    assert isinstance(dist, dict)
    assert "ema_trend" in dist


def test_condition_matrix_shape_and_dtypes():
    cfg = Config()
    df = make_synthetic_dataframe(n=600, tf="1h", seed=7)
    funnel, mat = instrument_signals(cfg, df)
    assert list(mat.columns) == ["ema_trend", "rsi_momentum", "atr_valid",
                                  "sr_room", "rr_ok", "score_ok"]
    assert mat.dtypes.apply(lambda d: d == bool).all()


# ---- 23.5 Correlation / bottleneck ----

def test_correlation_matrix_symmetric():
    cfg = Config()
    df = make_synthetic_dataframe(n=600, tf="1h", seed=8)
    _, mat = instrument_signals(cfg, df)
    corr = condition_correlation_matrix(mat)
    for a in corr.columns:
        for b in corr.columns:
            assert abs(corr.loc[a, b] - corr.loc[b, a]) < 1e-9


def test_bottleneck_returns_sorted_pass_rates():
    cfg = Config()
    df = make_synthetic_dataframe(n=600, tf="1h", seed=9)
    _, mat = instrument_signals(cfg, df)
    bn = bottleneck_analysis(mat)
    rates = list(bn.values())
    assert rates == sorted(rates)
    assert all(0.0 <= r <= 1.0 for r in rates)


# ---- 23.7 Counterfactual ----

def test_counterfactual_progressive_is_diagnostic_only():
    cfg = Config()
    df = make_synthetic_dataframe(n=1500, tf="1h", seed=10)
    sim = Simulator()
    results = counterfactual_progressive(cfg, df, sim)
    assert len(results) == 4
    for r in results:
        assert r.diagnostic_only is True
        assert r.not_deployable is True
        assert r.not_oos_validated is True
    a = next(r for r in results if r.label.startswith("A"))
    d = next(r for r in results if r.label.startswith("D"))
    assert a.candidate_count >= d.candidate_count


# ---- 23.8 Multi-timeframe ----

def test_multi_timeframe_diagnostic():
    cfg = Config()
    df1h = make_synthetic_dataframe(n=800, tf="1h", seed=11)
    df4h = make_synthetic_dataframe(n=400, tf="4h", seed=12)
    sim = Simulator()
    out = multi_timeframe_diagnostic(cfg, {"1h": df1h, "4h": df4h, "1d": None}, sim)
    assert "1h" in out and "4h" in out and "1d" in out
    assert out["1d"]["status"] == "INSUFFICIENT_DATA"
    assert out["1h"]["candles"] == 800


# ---- 23.9 Regime signal frequency ----

def test_per_regime_signal_frequency():
    cfg = Config()
    df = make_synthetic_dataframe(n=1500, tf="1h", seed=13)
    rdf = per_regime_signal_frequency(cfg, df)
    assert "candle" in rdf.columns and "candidate" in rdf.columns
    assert rdf["candle"].sum() == 1500
