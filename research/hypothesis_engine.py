"""Hypothesis tracking: every experiment is logged, including failures.

The research loop must record every hypothesis, its parameters, its OOS result,
and whether it was promoted or rejected. This is the audit trail that prevents
"selecting only the best historical result" and supports multiple-testing awareness.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import os


@dataclass
class HypothesisResult:
    hypothesis_id: str
    name: str
    family: str  # trend|momentum|breakout|pullback|mean_reversion|...
    parameters: Dict[str, Any]
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    oos_metrics: Dict[str, float]
    promoted: bool
    rejection_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HypothesisLog:
    """Append-only log of every hypothesis tested."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.records: List[HypothesisResult] = []
        self._counter = 0

    def new_id(self) -> str:
        self._counter += 1
        return f"H{self._counter:04d}"

    def record(self, result: HypothesisResult) -> None:
        self.records.append(result)
        if self.path:
            self._persist(result)

    def _persist(self, r: HypothesisResult) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(r), default=str) + "\n")

    def count(self) -> int:
        return len(self.records)

    def promoted(self) -> List[HypothesisResult]:
        return [r for r in self.records if r.promoted]

    def rejected(self) -> List[HypothesisResult]:
        return [r for r in self.records if not r.promoted]

    def families(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.records:
            out[r.family] = out.get(r.family, 0) + 1
        return out
