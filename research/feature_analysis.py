"""Feature analysis: importance / correlation / look-ahead leakage checks."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def feature_target_correlation(features: pd.DataFrame, target: pd.Series,
                               cols: List[str]) -> Dict[str, float]:
    """Pearson correlation between each feature and a forward target.

    The caller is responsible for defining `target` correctly (e.g. forward
    return shifted so there is no look-ahead). This function only measures.
    """
    out: Dict[str, float] = {}
    t = target.astype(float)
    for c in cols:
        if c not in features.columns:
            out[c] = float("nan")
            continue
        x = features[c].astype(float)
        m = x.notna() & t.notna()
        if m.sum() < 5:
            out[c] = float("nan")
            continue
        out[c] = float(np.corrcoef(x[m].to_numpy(), t[m].to_numpy())[0, 1])
    return out


def look_ahead_leakage_check(features: pd.DataFrame, target: pd.Series,
                              cols: List[str], max_lag: int = 5) -> Dict[str, float]:
    """For each feature, check correlation with target at lags 0..max_lag.

    A robust feature should have stable or decaying correlation; a sharp
    change at lag 0 vs negative lags suggests the target was used to build
    the feature (leakage). Returns the max absolute change in corr across lags.
    """
    out: Dict[str, float] = {}
    for c in cols:
        if c not in features.columns:
            out[c] = float("nan")
            continue
        x = features[c].astype(float)
        corrs = []
        for lag in range(-max_lag, 1):  # negative lags = target AFTER feature
            shifted = target.shift(-lag) if lag < 0 else target
            m = x.notna() & shifted.notna()
            if m.sum() < 5:
                corrs.append(np.nan)
                continue
            corrs.append(np.corrcoef(x[m].to_numpy(), shifted[m].to_numpy())[0, 1])
        corrs = [v for v in corrs if not np.isnan(v)]
        if len(corrs) < 2:
            out[c] = 0.0
            continue
        out[c] = float(np.max(np.abs(np.diff(corrs))))
    return out


def feature_count_penalty(n_features: int, baseline: int = 10) -> float:
    """Penalty for overfitting risk as feature count grows (0..1, 1 = no penalty)."""
    if n_features <= baseline:
        return 1.0
    return max(0.0, 1.0 - 0.03 * (n_features - baseline))
