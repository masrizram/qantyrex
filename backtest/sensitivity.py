"""Sensitivity analysis: perturb strategy parameters and measure metric stability."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np


@dataclass
class SensitivityReport:
    base_metrics: Dict[str, float]
    perturbed_results: List[Dict[str, float]] = field(default_factory=list)
    stability: Dict[str, float] = field(default_factory=dict)  # param -> fraction of viable perturbations


def analyze_sensitivity(
    base_cfg: dict,
    perturbations: Dict[str, List[float]],
    run_fn: Callable[[dict], Dict[str, float]],
    *,
    viability: Callable[[Dict[str, float]], bool] = lambda m: m.get("expectancy", 0) > 0 and m.get("profit_factor", 0) > 1.0,
) -> SensitivityReport:
    """`run_fn(cfg_dict) -> metrics_dict`. A robust strategy remains viable
    across a neighborhood of each parameter; we report the fraction of
    perturbations that remain viable."""
    base_metrics = run_fn(base_cfg)
    results: List[Dict[str, float]] = []
    stability: Dict[str, float] = {}
    for param, values in perturbations.items():
        viable = 0
        for v in values:
            cfg = dict(base_cfg)
            cfg[param] = v
            m = run_fn(cfg)
            results.append({"param": param, "value": v, **m})
            if viability(m):
                viable += 1
        stability[param] = viable / max(1, len(values))
    return SensitivityReport(base_metrics=base_metrics,
                             perturbed_results=results,
                             stability=stability)


def overall_stability(report: SensitivityReport) -> float:
    """Mean fraction of viable perturbations across all parameters (0..1)."""
    if not report.stability:
        return 0.0
    return float(np.mean(list(report.stability.values())))
