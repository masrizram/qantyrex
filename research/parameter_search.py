"""Parameter search with multiple-testing awareness.

We grid (or randomly sample) a *small* parameter neighborhood and count every
evaluation. The family-wise error rate grows with the number of tests; the
caller reports this count so selection is not "select the best of N silently".
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import random


@dataclass
class ParameterSpec:
    name: str
    values: Sequence  # discrete grid for this parameter


def grid_search(specs: List[ParameterSpec]) -> List[Dict]:
    """Full cartesian product. Use sparingly (exponential)."""
    names = [s.name for s in specs]
    grids = [list(s.values) for s in specs]
    return [dict(zip(names, combo)) for combo in product(*grids)]


def random_search(specs: List[ParameterSpec], n: int = 50,
                  seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    out: List[Dict] = []
    for _ in range(n):
        combo = {s.name: rng.choice(list(s.values)) for s in specs}
        out.append(combo)
    return out


@dataclass
class SearchReport:
    total_evaluations: int
    best_params: Dict
    best_score: float
    all_scores: List[Tuple[Dict, float]]


def run_search(params: List[ParameterSpec], objective: Callable[[Dict], float],
               mode: str = "grid", n: int = 50, seed: int = 0,
               maximize: bool = True) -> SearchReport:
    if mode == "grid":
        combos = grid_search(params)
    elif mode == "random":
        combos = random_search(params, n=n, seed=seed)
    else:
        raise ValueError(f"Unknown search mode {mode!r}")
    scored: List[Tuple[Dict, float]] = []
    for c in combos:
        score = float(objective(c))
        scored.append((c, score))
    if not scored:
        return SearchReport(0, {}, float("nan"), [])
    best = max(scored, key=lambda x: x[1]) if maximize else min(scored, key=lambda x: x[1])
    return SearchReport(
        total_evaluations=len(scored),
        best_params=best[0],
        best_score=best[1],
        all_scores=scored,
    )


def bonferroni_alpha(alpha: float, n_tests: int) -> float:
    """Return the Bonferroni-adjusted significance level."""
    if n_tests <= 0:
        return alpha
    return max(0.0, alpha / n_tests)
