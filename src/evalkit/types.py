"""Core data types for suites, cases, and results."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Usage:
    """Token counts and derived dollar cost for one model call.

    cost_usd is None when the model is absent from the pricing table. None means
    "unknown", which is deliberately NOT the same as 0.0 ("free"). A cost gate
    must not be satisfied by a number nobody verified.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def priced(self) -> bool:
        return self.cost_usd is not None

    def __add__(self, other: "Usage") -> "Usage":
        if self.cost_usd is None or other.cost_usd is None:
            cost = None
        else:
            cost = round(self.cost_usd + other.cost_usd, 8)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=cost,
        )


@dataclass
class Completion:
    """What a provider returned for a single prompt."""

    text: str
    usage: Usage
    latency_ms: float
    model: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class AssertionResult:
    kind: str
    passed: bool
    detail: str = ""
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseResult:
    """Outcome of one case against one model."""

    case_id: str
    model: str
    passed: bool
    output: str
    assertions: list[AssertionResult]
    usage: Usage
    latency_ms: float
    error: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Fraction of assertions that passed (1.0 when there are none)."""
        if not self.assertions:
            return 1.0 if self.error is None else 0.0
        return sum(1 for a in self.assertions if a.passed) / len(self.assertions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model": self.model,
            "passed": self.passed,
            "output": self.output,
            "assertions": [a.to_dict() for a in self.assertions],
            "usage": asdict(self.usage),
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "tags": self.tags,
            "score": round(self.score, 4),
        }


@dataclass
class RunResult:
    """A full suite execution — the artifact written to disk and diffed."""

    suite: str
    results: list[CaseResult]
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    # -- aggregates -------------------------------------------------------
    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.results) / self.total if self.total else 0.0

    @property
    def total_cost(self) -> float:
        """Sum of known costs. Unpriced results contribute 0; see `unpriced`."""
        return round(sum(r.usage.cost_usd or 0.0 for r in self.results), 6)

    @property
    def unpriced(self) -> list[str]:
        """Models whose cost could not be computed — reported, never guessed."""
        return sorted({r.model for r in self.results if not r.usage.priced})

    @property
    def fully_priced(self) -> bool:
        return not self.unpriced

    @property
    def total_tokens(self) -> int:
        return sum(r.usage.input_tokens + r.usage.output_tokens for r in self.results)

    def latency_p(self, pct: float) -> float:
        """Percentile latency in ms. pct is 0-100."""
        if not self.results:
            return 0.0
        vals = sorted(r.latency_ms for r in self.results)
        if len(vals) == 1:
            return round(vals[0], 2)
        pos = (pct / 100) * (len(vals) - 1)
        low, high = int(pos), min(int(pos) + 1, len(vals) - 1)
        frac = pos - low
        return round(vals[low] + (vals[high] - vals[low]) * frac, 2)

    def by_model(self) -> dict[str, list[CaseResult]]:
        out: dict[str, list[CaseResult]] = {}
        for r in self.results:
            out.setdefault(r.model, []).append(r)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "meta": self.meta,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": round(self.pass_rate, 4),
                "mean_score": round(self.mean_score, 4),
                "total_cost_usd": self.total_cost,
                "fully_priced": self.fully_priced,
                "unpriced_models": self.unpriced,
                "total_tokens": self.total_tokens,
                "latency_p50_ms": self.latency_p(50),
                "latency_p95_ms": self.latency_p(95),
            },
            "results": [r.to_dict() for r in self.results],
        }
