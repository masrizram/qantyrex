"""Tests for research engine: hypothesis log, parameter search, optimizer, selector, feature analysis."""
import math
import os
import pytest
import numpy as np
import pandas as pd

from trading_bot.research.hypothesis_engine import HypothesisLog, HypothesisResult
from trading_bot.research.parameter_search import (
    ParameterSpec, grid_search, random_search, run_search, bonferroni_alpha
)
from trading_bot.research.optimizer import StrategyScore, rank_strategies, select_best, composite_score
from trading_bot.research.strategy_selector import evaluate_readiness
from trading_bot.research.feature_analysis import (
    feature_target_correlation, look_ahead_leakage_check, feature_count_penalty
)
from trading_bot.core.enums import LiveReadiness


def test_hypothesis_log_append_only(tmp_path):
    p = tmp_path / "hyp.jsonl"
    log = HypothesisLog(str(p))
    hid = log.new_id()
    r = HypothesisResult(
        hypothesis_id=hid, name="trend_ema", family="trend",
        parameters={"fast": 50, "slow": 200},
        train_metrics={"pf": 1.4}, validation_metrics={"pf": 1.2},
        oos_metrics={"pf": 0.9}, promoted=False, rejection_reason="oos_pf_le_1",
    )
    log.record(r)
    assert log.count() == 1
    assert len(log.rejected()) == 1 and len(log.promoted()) == 0
    assert os.path.exists(str(p))
    # idempotent counter increments
    assert log.new_id() == "H0002"


def test_grid_search_combinations():
    specs = [ParameterSpec("a", [1, 2]), ParameterSpec("b", [10, 20, 30])]
    combos = grid_search(specs)
    assert len(combos) == 6
    assert {"a", "b"} == set(combos[0].keys())


def test_random_search_count():
    specs = [ParameterSpec("a", [1, 2, 3])]
    combos = random_search(specs, n=10, seed=1)
    assert len(combos) == 10
    assert all(c["a"] in (1, 2, 3) for c in combos)


def test_run_search_maximize_and_minimize():
    specs = [ParameterSpec("x", [0, 1, 2, 3])]
    rep = run_search(specs, lambda c: c["x"] ** 2, mode="grid", maximize=True)
    assert rep.total_evaluations == 4
    assert rep.best_params == {"x": 3} and rep.best_score == 9
    rep_min = run_search(specs, lambda c: c["x"] ** 2, mode="grid", maximize=False)
    assert rep_min.best_params == {"x": 0}


def test_bonferroni():
    assert bonferroni_alpha(0.05, 10) == 0.005
    assert bonferroni_alpha(0.05, 0) == 0.05


def test_composite_score_ranking_prefers_oos():
    good = StrategyScore("good", oos_expectancy=0.1, oos_profit_factor=1.5,
                         max_drawdown=0.1, wf_consistency=0.8, mc_survival=0.9,
                         param_stability=0.9, regime_robustness=0.8,
                         execution_robustness=0.9)
    bad = StrategyScore("bad", oos_expectancy=-0.05, oos_profit_factor=0.9,
                        max_drawdown=0.4, wf_consistency=0.3, mc_survival=0.4,
                        param_stability=0.4, regime_robustness=0.3,
                        execution_robustness=0.4)
    ranked = rank_strategies([bad, good])
    assert ranked[0].name == "good"
    assert ranked[0].composite > ranked[1].composite
    best = select_best([bad, good])
    assert best.name == "good"


def test_selector_rejects_when_oos_negative():
    s = StrategyScore("x", oos_expectancy=-0.02, oos_profit_factor=0.95,
                      max_drawdown=0.1, wf_consistency=0.9, mc_survival=0.95,
                      param_stability=0.9, regime_robustness=0.9,
                      execution_robustness=0.9)
    composite_score(s)
    d = evaluate_readiness(s)
    assert d.readiness == LiveReadiness.REJECTED
    assert any("oos_expectancy" in r for r in d.reasons)


def test_selector_paper_then_micro_then_production():
    s = StrategyScore("x", oos_expectancy=0.05, oos_profit_factor=1.4,
                      max_drawdown=0.1, wf_consistency=0.8, mc_survival=0.9,
                      param_stability=0.9, regime_robustness=0.8,
                      execution_robustness=0.9)
    composite_score(s)
    assert evaluate_readiness(s).readiness == LiveReadiness.PAPER_TRADING
    assert evaluate_readiness(s, paper_verified=True).readiness == LiveReadiness.MICRO_LIVE
    assert evaluate_readiness(s, paper_verified=True, micro_live_verified=True).readiness == LiveReadiness.PRODUCTION_CANDIDATE


def test_feature_target_correlation_basic():
    f = pd.DataFrame({"a": np.arange(100.0), "b": -np.arange(100.0)})
    t = pd.Series(np.arange(100.0))
    c = feature_target_correlation(f, t, ["a", "b", "missing"])
    assert c["a"] > 0.99
    assert c["b"] < -0.99
    assert math.isnan(c["missing"])


def test_look_ahead_leakage_check_detects_clean_feature():
    # feature that is a clean function of past returns only -> small cross-lag change
    n = 500
    rng = np.random.default_rng(0)
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    feat = pd.Series(np.log(pd.Series(price)).diff().rolling(10).std())
    target = pd.Series(np.log(pd.Series(price)).diff().shift(-1))  # next-bar return
    out = look_ahead_leakage_check(pd.DataFrame({"vol": feat}), target, ["vol"], max_lag=3)
    assert out["vol"] < 0.5  # not a sharp jump


def test_feature_count_penalty():
    assert feature_count_penalty(5) == 1.0
    assert feature_count_penalty(10) == 1.0
    assert feature_count_penalty(20) < 1.0
    assert feature_count_penalty(50) < feature_count_penalty(20)
