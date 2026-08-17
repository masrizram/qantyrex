"""Signal-frequency diagnostic engine (Phase 23.4-23.9).

DIAGNOSTIC ONLY. Does NOT modify the baseline strategy, does NOT optimize,
does NOT produce deployable candidates. It instruments every signal condition
to identify which filter causes signal collapse.

Produces:
- condition funnel (per-condition pass counts + cumulative rates)
- condition correlation matrix (bottleneck identification)
- rejection-reason distribution (explicit per-rejection)
- counterfactual progressive-filter results (DIAGNOSTIC_ONLY, NOT deployable)
- multi-timeframe diagnostics
- per-regime signal frequency
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..core.enums import Side, TrendState, RegimeState, RegimeAction
from .indicators import atr, rsi, macd, adx, ema
from .momentum import MomentumConfig, buy_momentum, sell_momentum
from .regime import classify_regime, RegimeConfig, action_for
from .support_resistance import SRConfig, add_support_resistance, room_for_tp
from .trend import classify_trend
from .volatility import VolatilityConfig, volatility_status
from .scoring import score_signal


CONDITIONS = ["ema_trend", "rsi_momentum", "atr_valid", "sr_room", "rr_ok", "score_ok"]


@dataclass
class FunnelRow:
    condition: str
    pass_count: int
    cumulative_rate: float  # pass / total
    conditional_rate: float  # pass / previous_pass
    rejected_reasons: Dict[str, int] = field(default_factory=dict)


@dataclass
class FunnelReport:
    total_candles: int
    rows: List[FunnelRow] = field(default_factory=list)
    final_buy: int = 0
    final_sell: int = 0
    final_signals: int = 0

    def summary(self) -> str:
        lines = [f"SIGNAL FREQUENCY FUNNEL  total_candles={self.total_candles}"]
        prev = self.total_candles
        for r in self.rows:
            lines.append(f"  {r.condition:<14} pass={r.pass_count:>6}  "
                         f"cum_rate={r.cumulative_rate:.3f}  cond_rate={r.conditional_rate:.3f}"
                         + (f"  top_reject={list(r.rejected_reasons.items())[:3]}" if r.rejected_reasons else ""))
        lines.append(f"  FINAL_BUY={self.final_buy}  FINAL_SELL={self.final_sell}  TOTAL_SIGNALS={self.final_signals}")
        return "\n".join(lines)


@dataclass
class CounterfactualResult:
    label: str
    candidate_count: int
    trade_count: int  # trades if executed by simulator (filled downstream)
    win_rate: float
    profit_factor: float
    expectancy: float
    diagnostic_only: bool = True
    not_deployable: bool = True
    not_oos_validated: bool = True


def _is_trend_bullish(trend: object) -> bool:
    v = trend.value if isinstance(trend, TrendState) else str(trend)
    return v in (TrendState.STRONG_BULLISH.value, TrendState.BULLISH.value)


def _is_trend_bearish(trend: object) -> bool:
    v = trend.value if isinstance(trend, TrendState) else str(trend)
    return v in (TrendState.STRONG_BEARISH.value, TrendState.BEARISH.value)


def instrument_signals(cfg: Config, df: pd.DataFrame,
                       df_trend: Optional[pd.DataFrame] = None,
                       spread_percent: float = 0.0,
                       liquidity_ok: bool = True) -> Tuple[FunnelReport, pd.DataFrame]:
    """Instrument every condition per candle. Returns the funnel + a per-candle
    boolean condition matrix (for correlation analysis)."""
    # build features the same way the signal engine does (no look-ahead)
    feat = classify_trend(df, cfg.ema_fast, cfg.ema_slow)
    mcfg = MomentumConfig(rsi_period=cfg.rsi_period, rsi_oversold=cfg.rsi_oversold,
                          rsi_overbought=cfg.rsi_overbought, adx_period=cfg.adx_period,
                          adx_min=cfg.adx_min)
    feat["rsi"] = rsi(feat["close"], cfg.rsi_period)
    macd_line, signal_line, _ = macd(feat["close"])
    feat["macd"] = macd_line
    feat["macd_signal"] = signal_line
    feat["adx"] = adx(feat, cfg.adx_period)
    feat["rsi_prev"] = feat["rsi"].shift(1)
    feat["atr"] = atr(feat, cfg.atr_period)
    feat["atr_percent"] = feat["atr"] / feat["close"] * 100.0
    vcfg = VolatilityConfig(atr_period=cfg.atr_period, atr_min_percent=cfg.atr_min_percent,
                            atr_max_percent=cfg.atr_max_percent)
    feat = classify_regime(feat, RegimeConfig(adx_period=cfg.adx_period, atr_period=cfg.atr_period))
    feat = add_support_resistance(feat, SRConfig())

    n = len(feat)
    _COL_IDX = {c: i for i, c in enumerate(CONDITIONS)}
    _cond = np.zeros((n, len(CONDITIONS)), dtype=bool)
    reasons: Dict[str, Dict[str, int]] = {c: {} for c in CONDITIONS}
    final_buy = 0
    final_sell = 0

    def _set(i, col, val):
        _cond[i, _COL_IDX[col]] = bool(val)

    for i in range(n):
        row = feat.iloc[i]
        if pd.isna(row.get("ema_slow")) or pd.isna(row.get("rsi")) or pd.isna(row.get("adx")):
            reasons["ema_trend"]["warmup"] = reasons["ema_trend"].get("warmup", 0) + 1
            continue
        # 1) EMA trend (bullish OR bearish, not neutral)
        trend = row["trend_state"]
        trend = trend if isinstance(trend, TrendState) else TrendState(str(trend))
        trend_bull = _is_trend_bullish(trend)
        trend_bear = _is_trend_bearish(trend)
        ema_ok = trend_bull or trend_bear
        _set(i, "ema_trend", ema_ok)
        if not ema_ok:
            reasons["ema_trend"]["neutral"] = reasons["ema_trend"].get("neutral", 0) + 1
            continue
        # 2) RSI/momentum
        side = Side.BUY if trend_bull else Side.SELL
        if side == Side.BUY:
            m_ok, m_reason = buy_momentum(row, mcfg)
        else:
            m_ok, m_reason = sell_momentum(row, mcfg)
        _set(i, "rsi_momentum", m_ok)
        if not m_ok:
            reasons["rsi_momentum"][m_reason] = reasons["rsi_momentum"].get(m_reason, 0) + 1
            continue
        # 3) ATR valid
        vlabel, vok = volatility_status(row, vcfg)
        _set(i, "atr_valid", vok)
        if not vok:
            reasons["atr_valid"][vlabel] = reasons["atr_valid"].get(vlabel, 0) + 1
            continue
        # 4) S/R room for TP
        sl = row["close"] - 1.5 * row["atr"] if side == Side.BUY else row["close"] + 1.5 * row["atr"]
        ns = row.get("nearest_support", np.nan)
        nr = row.get("nearest_resistance", np.nan)
        if side == Side.BUY and (pd.isna(sl) or sl >= row["close"]):
            _set(i, "sr_room", False)
            reasons["sr_room"]["sl_invalid"] = reasons["sr_room"].get("sl_invalid", 0) + 1
            continue
        if side == Side.SELL and (pd.isna(sl) or sl <= row["close"]):
            _set(i, "sr_room", False)
            reasons["sr_room"]["sl_invalid"] = reasons["sr_room"].get("sl_invalid", 0) + 1
            continue
        risk = abs(row["close"] - sl)
        tp = row["close"] + cfg.min_rr * risk if side == Side.BUY else row["close"] - cfg.min_rr * risk
        room_ok, room_reason = room_for_tp(row["close"], tp, side.value, nr, ns)
        _set(i, "sr_room", room_ok)
        if not room_ok:
            reasons["sr_room"][room_reason] = reasons["sr_room"].get(room_reason, 0) + 1
            continue
        # 5) RR >= min
        rr = cfg.min_rr
        rr_ok = rr >= cfg.min_rr
        _set(i, "rr_ok", rr_ok)
        if not rr_ok:
            reasons["rr_ok"]["rr_below_min"] = reasons["rr_ok"].get("rr_below_min", 0) + 1
            continue
        # 6) score >= 75
        support_strength = int(row.get("support_strength", 0) or 0)
        resistance_strength = int(row.get("resistance_strength", 0) or 0)
        trend_tf = trend
        sb = score_signal(
            side=side, trend=trend, trend_tf=trend, entry_tf_aligned=True,
            momentum_passes=True, adx=float(row.get("adx", 0) or 0), adx_min=cfg.adx_min,
            support_strength=support_strength, resistance_strength=resistance_strength,
            room_ok=True, volatility_label=vlabel, volatility_ok=vok,
            spread_percent=spread_percent, max_spread=cfg.max_spread_percent,
            liquidity_ok=liquidity_ok, rr=rr, min_rr=cfg.min_rr,
            latency_ms=None, min_score=cfg.min_signal_score,
        )
        _set(i, "score_ok", sb.passes)
        if not sb.passes:
            reasons["score_ok"][f"score_{sb.total:.0f}"] = reasons["score_ok"].get(f"score_{sb.total:.0f}", 0) + 1
            continue
        # final signal
        if side == Side.BUY:
            final_buy += 1
        else:
            final_sell += 1

    cond_matrix = pd.DataFrame(_cond, columns=CONDITIONS, index=feat.index)

    # build funnel rows
    funnel = FunnelReport(total_candles=n, final_buy=final_buy,
                          final_sell=final_sell, final_signals=final_buy + final_sell)
    prev_count = n
    for c in CONDITIONS:
        pc = int(cond_matrix[c].sum())
        cum_rate = pc / n if n else 0.0
        cond_rate = pc / prev_count if prev_count else 0.0
        funnel.rows.append(FunnelRow(condition=c, pass_count=pc,
                                     cumulative_rate=cum_rate, conditional_rate=cond_rate,
                                     rejected_reasons=dict(reasons[c])))
        prev_count = pc
    return funnel, cond_matrix


def condition_correlation_matrix(cond_matrix: pd.DataFrame) -> pd.DataFrame:
    """Pairwise co-occurrence (Jaccard) of condition passes. Identifies filters
    that almost always fail together (redundant) or that dominate as bottlenecks."""
    cols = list(cond_matrix.columns)
    n = len(cond_matrix)
    m = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            both = int((cond_matrix[a] & cond_matrix[b]).sum())
            either = int((cond_matrix[a] | cond_matrix[b]).sum())
            m.loc[a, b] = both / either if either else 0.0
    return m.astype(float)


def bottleneck_analysis(cond_matrix: pd.DataFrame) -> Dict[str, float]:
    """Pass-rate per condition. The lowest pass-rate condition is the dominant
    bottleneck. Diagnostic only — does NOT suggest changes."""
    out = {}
    n = len(cond_matrix)
    for c in cond_matrix.columns:
        out[c] = float(cond_matrix[c].sum()) / n if n else 0.0
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def rejection_reason_distribution(funnel: FunnelReport) -> Dict[str, Dict[str, int]]:
    return {row.condition: dict(row.rejected_reasons) for row in funnel.rows}


def counterfactual_progressive(cfg: Config, df: pd.DataFrame,
                              simulator) -> List[CounterfactualResult]:
    """Run diagnostic-only progressive filter variants. NOT deployable.

    A: EMA + RSI
    B: EMA + RSI + ATR
    C: EMA + RSI + ATR + S/R
    D: full baseline

    Each variant uses progressively more filters. Trades are executed by the
    provided `simulator` to get win_rate/PF/expectancy. These results are
    DIAGNOSTIC_ONLY — never promoted or optimized.
    """
    from ..core.models import Signal, TrendState as TS, RegimeState as RS, RegimeAction as RA
    feat = classify_trend(df, cfg.ema_fast, cfg.ema_slow)
    mcfg = MomentumConfig(rsi_period=cfg.rsi_period, rsi_oversold=cfg.rsi_oversold,
                          rsi_overbought=cfg.rsi_overbought, adx_period=cfg.adx_period,
                          adx_min=cfg.adx_min)
    feat["rsi"] = rsi(feat["close"], cfg.rsi_period)
    ml, sl, _ = macd(feat["close"])
    feat["macd"] = ml; feat["macd_signal"] = sl
    feat["adx"] = adx(feat, cfg.adx_period)
    feat["rsi_prev"] = feat["rsi"].shift(1)
    feat["atr"] = atr(feat, cfg.atr_period)
    feat["atr_percent"] = feat["atr"] / feat["close"] * 100.0
    vcfg = VolatilityConfig(atr_period=cfg.atr_period, atr_min_percent=cfg.atr_min_percent,
                            atr_max_percent=cfg.atr_max_percent)
    feat = classify_regime(feat, RegimeConfig(adx_period=cfg.adx_period, atr_period=cfg.atr_period))
    feat = add_support_resistance(feat, SRConfig())

    variants = {"A_EMA_RSI": ["ema", "rsi"],
                "B_EMA_RSI_ATR": ["ema", "rsi", "atr"],
                "C_EMA_RSI_ATR_SR": ["ema", "rsi", "atr", "sr"],
                "D_full_baseline": ["ema", "rsi", "atr", "sr", "score"]}
    results: List[CounterfactualResult] = []
    for label, filters in variants.items():
        signals = []
        for i in range(len(feat)):
            row = feat.iloc[i]
            if pd.isna(row.get("ema_slow")) or pd.isna(row.get("rsi")):
                continue
            trend = row["trend_state"]
            trend = trend if isinstance(trend, TS) else TS(str(trend))
            if trend == TS.NEUTRAL:
                continue
            side = Side.BUY if _is_trend_bullish(trend) else Side.SELL
            if side == Side.BUY:
                m_ok, _ = buy_momentum(row, mcfg)
            else:
                m_ok, _ = sell_momentum(row, mcfg)
            if "rsi" in filters and not m_ok:
                continue
            if "atr" in filters:
                _, vok = volatility_status(row, vcfg)
                if not vok:
                    continue
            sl = row["close"] - 1.5 * row["atr"] if side == Side.BUY else row["close"] + 1.5 * row["atr"]
            if pd.isna(sl):
                continue
            risk = abs(row["close"] - sl)
            if risk <= 0:
                continue
            tp = row["close"] + cfg.min_rr * risk if side == Side.BUY else row["close"] - cfg.min_rr * risk
            ns = row.get("nearest_support", np.nan)
            nr = row.get("nearest_resistance", np.nan)
            if "sr" in filters:
                room_ok, _ = room_for_tp(row["close"], tp, side.value, nr, ns)
                if not room_ok:
                    continue
            if "score" in filters:
                # regime must approve for the full baseline
                action = row.get("regime_action", "NO_TRADE")
                if action != RA.TRADE.value:
                    continue
                sb = score_signal(
                    side=side, trend=trend, trend_tf=trend, entry_tf_aligned=True,
                    momentum_passes=True, adx=float(row.get("adx", 0) or 0),
                    adx_min=cfg.adx_min, support_strength=int(row.get("support_strength", 0) or 0),
                    resistance_strength=int(row.get("resistance_strength", 0) or 0),
                    room_ok=True, volatility_label="normal", volatility_ok=True,
                    spread_percent=0.0, max_spread=cfg.max_spread_percent,
                    liquidity_ok=True, rr=cfg.min_rr, min_rr=cfg.min_rr,
                    latency_ms=None, min_score=cfg.min_signal_score,
                )
                if not sb.passes:
                    continue
            signals.append(Signal(
                strategy_version=f"diag_{label}", symbol=cfg.symbol, side=side,
                entry=float(row["close"]), stop_loss=float(sl), take_profit=float(tp),
                rr=cfg.min_rr, score=80.0, trend=trend,
                regime=RS.STRONG_TREND, regime_action=RA.TRADE,
                rsi=float(row.get("rsi", 0) or 0), atr=float(row.get("atr", 0) or 0),
                atr_percent=float(row.get("atr_percent", 0) or 0),
                adx=float(row.get("adx", 0) or 0),
                ema_fast=float(row.get("ema_fast", 0) or 0),
                ema_slow=float(row.get("ema_slow", 0) or 0),
                spread_percent=0.0, timestamp=int(row["timestamp"]),
                features={"quantity": 1.0},
            ))
        sim_result = simulator.run(df, signals, features_for_atr=feat)
        m = sim_result.metrics
        results.append(CounterfactualResult(
            label=label, candidate_count=len(signals), trade_count=m.trade_count,
            win_rate=m.win_rate, profit_factor=m.profit_factor, expectancy=m.expectancy,
        ))
    return results


def per_regime_signal_frequency(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """For each coarse regime, count candles, candidate signals (post-EMA),
    accepted signals (post-score), and trades (via a dry run)."""
    funnel, cond_matrix = instrument_signals(cfg, df)
    # re-tag regimes
    cov_df = df.copy()
    from ..research.regime_coverage import classify_regime_coverage
    cov = classify_regime_coverage(cov_df, cfg.symbol)
    seg_map = sorted([(s.start_ts, s.end_ts, s.regime) for s in cov.segments])

    def _regime_at(ts):
        for st, en, rg in seg_map:
            if st <= ts <= en:
                return rg
        return "unknown"

    out_rows = []
    feat = classify_trend(df, cfg.ema_fast, cfg.ema_slow)
    for i in range(len(df)):
        rg = _regime_at(int(df["timestamp"].iloc[i]))
        ema_pass = bool(cond_matrix.iloc[i]["ema_trend"]) if i < len(cond_matrix) else False
        score_pass = bool(cond_matrix.iloc[i]["score_ok"]) if i < len(cond_matrix) else False
        out_rows.append({"regime": rg, "candle": 1, "candidate": int(ema_pass),
                          "accepted": int(score_pass)})
    rdf = pd.DataFrame(out_rows)
    return rdf.groupby("regime", as_index=False).sum()


def multi_timeframe_diagnostic(cfg: Config, df_by_tf: Dict[str, pd.DataFrame],
                               simulator) -> Dict[str, dict]:
    """Run the funnel + a dry trade count on each timeframe independently."""
    out: Dict[str, dict] = {}
    for tf, df in df_by_tf.items():
        if df is None or len(df) < 250:
            out[tf] = {"candles": len(df) if df is not None else 0, "signals": 0,
                       "trades": 0, "status": "INSUFFICIENT_DATA"}
            continue
        funnel, _ = instrument_signals(cfg, df)
        out[tf] = {
            "candles": funnel.total_candles,
            "signals": funnel.final_signals,
            "buy": funnel.final_buy, "sell": funnel.final_sell,
            "trades": funnel.final_signals,  # proxy; real trades need simulator
            "status": "INSUFFICIENT_SAMPLE" if funnel.final_signals < 50 else "ADEQUATE_SAMPLE",
        }
    return out
