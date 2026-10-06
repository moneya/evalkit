"""Run-to-run comparison — the regression gate, including cost.

The premise: a prompt change that improves quality 2% while making inference
4x more expensive is a regression. Most harnesses only diff quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Delta:
    label: str
    before: float
    after: float
    unit: str = ""
    higher_is_better: bool = True

    @property
    def abs_change(self) -> float:
        return self.after - self.before

    @property
    def pct_change(self) -> float | None:
        if self.before == 0:
            return None
        return (self.after - self.before) / abs(self.before) * 100

    @property
    def ratio(self) -> float | None:
        if self.before == 0:
            return None
        return self.after / self.before

    @property
    def improved(self) -> bool:
        return self.abs_change > 0 if self.higher_is_better else self.abs_change < 0

    def arrow(self) -> str:
        """Direction of *goodness*, not of the raw number."""
        if self.abs_change == 0:
            return "="
        return "^" if self.improved else "v"


@dataclass
class Comparison:
    baseline_label: str
    current_label: str
    deltas: list[Delta]
    newly_failing: list[str] = field(default_factory=list)
    newly_passing: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def load_run(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"run artifact not found: {p}")
    return json.loads(p.read_text())


def _key(r: dict[str, Any]) -> str:
    return f"{r['case_id']}@{r['model']}"


def compare(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    max_cost_increase_pct: float = 20.0,
    max_pass_rate_drop_pct: float = 0.0,
    max_p95_increase_pct: float = 50.0,
    allow_new_failures: bool = False,
) -> Comparison:
    """Diff two run artifacts and decide whether the change is acceptable."""
    bs, cs = baseline["summary"], current["summary"]

    deltas = [
        Delta("pass rate", bs["pass_rate"] * 100, cs["pass_rate"] * 100, "%", True),
        Delta("mean score", bs["mean_score"], cs["mean_score"], "", True),
        Delta("total cost", bs["total_cost_usd"], cs["total_cost_usd"], "$", False),
        Delta("total tokens", bs["total_tokens"], cs["total_tokens"], "tok", False),
        Delta("p50 latency", bs["latency_p50_ms"], cs["latency_p50_ms"], "ms", False),
        Delta("p95 latency", bs["latency_p95_ms"], cs["latency_p95_ms"], "ms", False),
    ]

    b_map = {_key(r): r for r in baseline["results"]}
    c_map = {_key(r): r for r in current["results"]}

    shared = b_map.keys() & c_map.keys()
    newly_failing = sorted(k for k in shared if b_map[k]["passed"] and not c_map[k]["passed"])
    newly_passing = sorted(k for k in shared if not b_map[k]["passed"] and c_map[k]["passed"])
    added = sorted(c_map.keys() - b_map.keys())
    removed = sorted(b_map.keys() - c_map.keys())

    violations: list[str] = []

    pr = deltas[0]
    if pr.abs_change < -abs(max_pass_rate_drop_pct):
        violations.append(
            f"pass rate dropped {abs(pr.abs_change):.1f}pp "
            f"({pr.before:.1f}% -> {pr.after:.1f}%), tolerance {max_pass_rate_drop_pct:.1f}pp"
        )

    cost = deltas[2]
    b_priced = bs.get("fully_priced", True)
    c_priced = cs.get("fully_priced", True)
    if not (b_priced and c_priced):
        missing = sorted(set(bs.get("unpriced_models", [])) | set(cs.get("unpriced_models", [])))
        violations.append(
            "cost gate cannot be evaluated: unpriced model(s) "
            f"{missing} — add them to the pricing table"
        )
    elif cost.pct_change is not None and cost.pct_change > max_cost_increase_pct:
        ratio = f" ({cost.ratio:.2f}x)" if cost.ratio else ""
        violations.append(
            f"cost rose {cost.pct_change:.1f}%{ratio} "
            f"(${cost.before:.6f} -> ${cost.after:.6f}), tolerance {max_cost_increase_pct:.0f}%"
        )
    elif b_priced and c_priced and cost.before == 0 and cost.after > 0:
        violations.append(f"cost rose from $0 to ${cost.after:.6f}")

    p95 = deltas[5]
    if p95.pct_change is not None and p95.pct_change > max_p95_increase_pct:
        violations.append(
            f"p95 latency rose {p95.pct_change:.1f}% "
            f"({p95.before:.0f}ms -> {p95.after:.0f}ms), tolerance {max_p95_increase_pct:.0f}%"
        )

    if newly_failing and not allow_new_failures:
        violations.append(f"{len(newly_failing)} case(s) newly failing: {newly_failing[:5]}")

    return Comparison(
        baseline_label=baseline.get("suite", "baseline"),
        current_label=current.get("suite", "current"),
        deltas=deltas,
        newly_failing=newly_failing,
        newly_passing=newly_passing,
        added=added,
        removed=removed,
        violations=violations,
    )
