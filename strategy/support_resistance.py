"""Support/resistance engine: swing points, zones, distance metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .indicators import swing_highs_lows


@dataclass
class SRConfig:
    swing_window: int = 3
    cluster_pct: float = 0.5  # levels within cluster_pct% of price are merged
    max_levels: int = 5


def add_support_resistance(df: pd.DataFrame, cfg: SRConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SRConfig()
    out = df.copy()
    is_sh, is_sl = swing_highs_lows(out, cfg.swing_window)
    out["is_swing_high"] = is_sh
    out["is_swing_low"] = is_sl
    # Running swing levels: for each bar, the list of prior confirmed swing highs/lows.
    sh_vals = out.loc[is_sh, "high"].to_numpy()
    sl_vals = out.loc[is_sl, "low"].to_numpy()
    sh_idx = np.where(is_sh.to_numpy())[0]
    sl_idx = np.where(is_sl.to_numpy())[0]

    supports: List[float] = []
    resistances: List[float] = []
    nearest_support = []
    nearest_resistance = []
    support_strength = []
    resistance_strength = []
    for i in range(len(out)):
        # add swing points that have been confirmed by the time we reach bar i
        # (swing at index j is confirmed at j + swing_window; we only use j <= i - 1
        # to guarantee no look-ahead through the right-side window)
        confirmed_sh = sh_idx[sh_idx <= i - cfg.swing_window]
        confirmed_sl = sl_idx[sl_idx <= i - cfg.swing_window]
        r_levels = _cluster(out["high"].to_numpy()[confirmed_sh], cfg)
        s_levels = _cluster(out["low"].to_numpy()[confirmed_sl], cfg)
        close_i = out["close"].iloc[i]
        # nearest resistance above price
        above = [lvl for lvl in r_levels if lvl > close_i]
        below = [lvl for lvl in s_levels if lvl < close_i]
        nr = min(above, key=lambda x: x - close_i) if above else np.nan
        ns = max(below, key=lambda x: close_i - x) if below else np.nan
        nearest_resistance.append(nr)
        nearest_support.append(ns)
        # strength = number of cluster members
        resistance_strength.append(sum(1 for lvl in r_levels if lvl > close_i and abs(lvl - nr) / close_i * 100 < cfg.cluster_pct) if above else 0)
        support_strength.append(sum(1 for lvl in s_levels if lvl < close_i and abs(lvl - ns) / close_i * 100 < cfg.cluster_pct) if below else 0)

    out["nearest_support"] = nearest_support
    out["nearest_resistance"] = nearest_resistance
    out["support_strength"] = support_strength
    out["resistance_strength"] = resistance_strength
    out["distance_to_support_pct"] = (out["close"] - out["nearest_support"]) / out["close"] * 100
    out["distance_to_resistance_pct"] = (out["nearest_resistance"] - out["close"]) / out["close"] * 100
    return out


def _cluster(levels: np.ndarray, cfg: SRConfig) -> List[float]:
    if len(levels) == 0:
        return []
    levels = np.sort(levels)
    clusters = [[levels[0]]]
    for v in levels[1:]:
        last = clusters[-1][-1]
        if abs(v - last) / last * 100 < cfg.cluster_pct:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    # cluster representative = mean of members
    return [float(np.mean(c)) for c in clusters][-cfg.max_levels:]


def room_for_tp(entry: float, tp: float, side: str, nearest_resistance: float,
                nearest_support: float, min_room_pct: float = 0.5) -> tuple[bool, str]:
    """Check whether market structure gives enough room for TP."""
    if side == "BUY":
        if pd.isna(nearest_resistance):
            return True, "no_resistance"
        room_pct = (nearest_resistance - tp) / entry * 100
        if room_pct < min_room_pct:
            return False, f"resistance_blocks_tp room={room_pct:.2f}%"
        return True, "ok"
    else:  # SELL
        if pd.isna(nearest_support):
            return True, "no_support"
        room_pct = (tp - nearest_support) / entry * 100
        if room_pct < min_room_pct:
            return False, f"support_blocks_tp room={room_pct:.2f}%"
        return True, "ok"
