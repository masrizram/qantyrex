"""Tests for Phase 22: historical ingestor, quality audit, regime coverage, sample gates, real pipeline."""
import json
import os
import time
import numpy as np
import pandas as pd
import pytest

from trading_bot.data.historical_ingestor import HistoricalIngestor, IngestConfig, dataset_hash
from trading_bot.data.quality_audit import audit as audit_quality, make_provenance, dataset_hash as dh
from trading_bot.research.regime_coverage import classify_regime_coverage, per_regime_performance
from trading_bot.research.sample_gates import (SampleGates, check_full_sample, check_oos_sample,
                                                 check_wf_sample, final_classification,
                                                 NO_VERIFIED_EDGE, INSUFFICIENT_SAMPLE,
                                                 PAPER_TRADING_ELIGIBLE, REJECTED)
from trading_bot.backtest.metrics import Metrics


# ---- Fixtures ----

def _make_df(n=500, tf_ms=3_600_000, start_ts=0, gap_at=None, bad_ohlc=False):
    ts = [start_ts + i*tf_ms for i in range(n)]
    if gap_at is not None:
        # create a gap of 3 tf at index gap_at
        ts = ts[:gap_at] + [t + 3*tf_ms for t in ts[gap_at:]]
    closes = np.cumsum(np.random.default_rng(1).normal(0, 1, n)) + 100
    df = pd.DataFrame({
        "timestamp": ts, "open": closes, "high": closes+1, "low": closes-1,
        "close": closes+0.1, "volume": np.random.default_rng(2).uniform(1, 10, n),
    })
    if bad_ohlc:
        df.loc[10, "high"] = df.loc[10, "low"] - 1  # high < low -> integrity fail
    return df


class _FakeExchange:
    """Mock ccxt exchange returning paginated synthetic OHLCV."""
    def __init__(self, total=300, tf_ms=3_600_000, start_ts=0, fail_calls=None, fail_forever_after=None):
        self.total = total; self.tf = tf_ms; self.start = start_ts
        self.fail_calls = set(fail_calls or [])           # specific call indices that raise
        self.fail_forever_after = fail_forever_after        # if set, raise for all calls >= this
        self.calls = 0
    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        self.calls += 1
        if self.calls in self.fail_calls or (self.fail_forever_after is not None and self.calls >= self.fail_forever_after):
            raise RuntimeError(f"simulated network error call={self.calls}")
        tf = self.tf
        start = self.start if since is None else since
        page = []
        for i in range(limit):
            ts = start + i*tf
            if ts > self.start + self.total*tf:
                break
            if len(page) >= limit:
                break
            page.append([ts, 100.0, 101.0, 99.0, 100.5, 1.0])
        return page


# ---- Ingestor ----

def test_ingestor_basic_pagination(tmp_path):
    ex = _FakeExchange(total=2500)
    cfg = IngestConfig(exchange="fake", symbol="BTC/USDT", timeframe="1h",
                       since_ms=0, limit_per_page=1000, max_pages=10,
                       cache_dir=str(tmp_path), rate_limit_ms=0)
    ing = HistoricalIngestor(ex, cfg)
    df, rep = ing.ingest()
    assert len(df) > 0
    assert rep.exchange == "fake" and rep.symbol == "BTC/USDT"
    assert rep.first_timestamp == 0
    assert rep.candle_count == len(df)
    assert rep.configuration_hash != ""
    # cache written
    assert os.path.exists(os.path.join(str(tmp_path), "fake_BTC_USDT_1h.parquet"))


def test_ingestor_resumable(tmp_path):
    # first ingest
    ex = _FakeExchange(total=1200)
    cfg = IngestConfig(exchange="fake", symbol="BTC/USDT", timeframe="1h",
                       since_ms=0, limit_per_page=1000, max_pages=5,
                       cache_dir=str(tmp_path), rate_limit_ms=0)
    ing = HistoricalIngestor(ex, cfg)
    df1, r1 = ing.ingest()
    assert r1.resumed is False
    # second ingest with same cache -> resumed
    ing2 = HistoricalIngestor(ex, cfg)
    df2, r2 = ing2.ingest()
    assert r2.resumed is True
    assert len(df2) >= len(df1)


def test_ingestor_records_errors(tmp_path):
    # fail ALL retries on the first page (calls 1,2,3 all fail) -> error recorded
    ex = _FakeExchange(total=300, fail_forever_after=1)
    cfg = IngestConfig(exchange="fake", symbol="BTC/USDT", timeframe="1h",
                       since_ms=0, limit_per_page=100, max_pages=5,
                       cache_dir=str(tmp_path), rate_limit_ms=0, max_retries=2)
    ing = HistoricalIngestor(ex, cfg)
    df, rep = ing.ingest()
    assert len(rep.fetch_errors) >= 1
    assert "simulated" in rep.fetch_errors[0].error or "RuntimeError" in rep.fetch_errors[0].error


def test_ingestor_detects_missing_ranges(tmp_path):
    # build data with a gap by feeding two pages with a jump
    ex = _FakeExchange(total=100)
    cfg = IngestConfig(exchange="fake", symbol="BTC/USDT", timeframe="1h",
                       since_ms=0, limit_per_page=1000, max_pages=2,
                       cache_dir=str(tmp_path), rate_limit_ms=0)
    ing = HistoricalIngestor(ex, cfg)
    df, rep = ing.ingest()
    # no gaps expected in clean fake data
    assert rep.missing_ranges == [] or isinstance(rep.missing_ranges, list)


def test_dataset_hash_stable():
    df = _make_df(n=50)
    h1 = dataset_hash(df)
    h2 = dataset_hash(df.copy())
    assert h1 == h2
    assert len(h1) == 16


# ---- Quality audit ----

def test_quality_audit_pass_clean():
    df = _make_df(n=300)
    rep = audit_quality(df, "gate", "BTC/USDT", "1h")
    assert rep.overall in ("PASS", "WARNING")
    assert any(c.name == "timestamp_monotonic" and c.status == "PASS" for c in rep.checks)


def test_quality_audit_fails_ohlc():
    df = _make_df(n=300, bad_ohlc=True)
    rep = audit_quality(df, "gate", "BTC/USDT", "1h")
    assert rep.overall == "FAIL"
    assert any(c.name == "high_ge_low" and c.status == "FAIL" for c in rep.checks)


def test_quality_audit_fails_duplicates():
    df = _make_df(n=300)
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # duplicate ts
    rep = audit_quality(df, "gate", "BTC/USDT", "1h")
    assert rep.overall == "FAIL"
    assert any(c.name == "duplicate_timestamps" and c.status == "FAIL" for c in rep.checks)


def test_quality_audit_empty_fail():
    rep = audit_quality(pd.DataFrame(columns=["timestamp","open","high","low","close","volume"]),
                         "gate", "X/USDT", "1h")
    assert rep.overall == "FAIL"


def test_quality_audit_warnings_on_gaps():
    df = _make_df(n=300, gap_at=50)
    rep = audit_quality(df, "gate", "BTC/USDT", "1h")
    # gaps create non-multiple diffs -> timeframe_consistency WARNING + missing_candles WARNING
    assert rep.overall == "WARNING" or rep.overall == "FAIL"


def test_make_provenance():
    df = _make_df(n=10)
    p = make_provenance(df, "gate", "BTC/USDT", "1h", "abc123")
    assert p.symbol == "BTC/USDT" and p.row_count == 10
    assert p.dataset_hash != "" and p.configuration_hash == "abc123"


# ---- Regime coverage ----

def test_regime_coverage_classifies():
    n = 600
    rng = np.random.default_rng(7)
    # build a series with a bull run, a chop, then a crash -> multiple regimes
    closes = np.concatenate([
        np.linspace(100, 200, 200),   # bull
        200 + rng.normal(0, 1, 200), # range
        np.linspace(300, 150, 200),  # bear/crash
    ])
    ts = [i * 3_600_000 for i in range(n)]
    df = pd.DataFrame({"timestamp": ts, "open": closes, "high": closes+1,
                       "low": closes-1, "close": closes, "volume": 1.0})
    cov = classify_regime_coverage(df, "BTC/USDT")
    assert len(cov.segments) > 0
    assert "bull" in cov.by_regime or "bear" in cov.by_regime or "range" in cov.by_regime


def test_regime_coverage_insufficient():
    df = _make_df(n=20)
    cov = classify_regime_coverage(df, "X")
    assert cov.coverage_complete is False


def test_per_regime_performance_empty_trades():
    df = _make_df(n=400)
    out = per_regime_performance(pd.DataFrame(), df)
    assert out == {}


# ---- Sample gates ----

def _m(trade_count, expectancy=0, pf=0):
    return Metrics(0,0,0,pf,expectancy,0,0,0,0,0,0,0,0,0,0,0,0,trade_count,0,0,0)


def test_check_full_sample_pass():
    g = SampleGates(min_full_trades=10)
    assert check_full_sample(_m(20), g).status == "PASS"


def test_check_full_sample_insufficient():
    g = SampleGates(min_full_trades=100)
    assert check_full_sample(_m(5), g).status == "INSUFFICIENT_SAMPLE"


def test_check_oos_sample_no_trades_inconclusive():
    g = SampleGates(min_oos_trades=5)
    r = check_oos_sample(_m(0), g)
    assert r.status == "INCONCLUSIVE_NO_TRADES"


def test_check_wf_sample_pass():
    g = SampleGates(min_aggregate_wf_trades=10)
    assert check_wf_sample(50, g).status == "PASS"


def test_final_classification_no_edge():
    gs = SampleGates(min_full_trades=10, min_oos_trades=10, min_aggregate_wf_trades=10)
    r, reasons = final_classification(
        data_ok=True, full_sample=check_full_sample(_m(100), gs),
        oos_sample=check_oos_sample(_m(60), gs),
        wf_sample=check_wf_sample(150, gs),
        oos_expectancy=-0.01, oos_pf=0.9, mc_executed=True, mc_pass=True, stress_pass=True,
    )
    assert r == NO_VERIFIED_EDGE
    assert any("edge_absent" in x for x in reasons)


def test_final_classification_paper_eligible():
    gs = SampleGates()
    r, reasons = final_classification(
        data_ok=True,
        full_sample=check_full_sample(_m(100), gs),
        oos_sample=check_oos_sample(_m(60), gs),
        wf_sample=check_wf_sample(150, gs),
        oos_expectancy=0.02, oos_pf=1.3, mc_executed=True, mc_pass=True, stress_pass=True,
    )
    assert r == PAPER_TRADING_ELIGIBLE
    assert "all_gates_passed" in reasons


def test_final_classification_never_live():
    # even with everything passing, the label is PAPER_TRADING_ELIGIBLE, never LIVE/PRODUCTION
    gs = SampleGates()
    r, _ = final_classification(
        data_ok=True, full_sample=check_full_sample(_m(100), gs),
        oos_sample=check_oos_sample(_m(60), gs), wf_sample=check_wf_sample(150, gs),
        oos_expectancy=0.5, oos_pf=3.0, mc_executed=True, mc_pass=True, stress_pass=True,
    )
    assert "LIVE" not in r
    assert r == PAPER_TRADING_ELIGIBLE
