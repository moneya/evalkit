"""Tests for the regression gate — the differentiating feature."""

from __future__ import annotations

from evalkit.compare import compare


def artifact(*, pass_rate=1.0, mean_score=1.0, cost=0.01, tokens=1000,
             p50=500.0, p95=900.0, results=None, suite="s"):
    return {
        "suite": suite,
        "summary": {
            "total": len(results or []),
            "passed": sum(1 for r in (results or []) if r["passed"]),
            "failed": sum(1 for r in (results or []) if not r["passed"]),
            "pass_rate": pass_rate,
            "mean_score": mean_score,
            "total_cost_usd": cost,
            "total_tokens": tokens,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
        },
        "results": results or [],
    }


def case(cid, passed, model="echo-1"):
    return {"case_id": cid, "model": model, "passed": passed, "output": "",
            "assertions": [], "usage": {}, "latency_ms": 0, "error": None,
            "tags": [], "score": 1.0 if passed else 0.0}


def test_identical_runs_pass():
    a = artifact(results=[case("x", True)])
    assert compare(a, artifact(results=[case("x", True)])).ok


def test_quality_up_but_cost_4x_is_a_regression():
    """The core thesis: better output at 4x the price still fails."""
    base = artifact(pass_rate=0.90, cost=0.010, results=[case("x", True)])
    new = artifact(pass_rate=0.92, cost=0.040, results=[case("x", True)])
    c = compare(base, new)
    assert not c.ok
    assert any("cost rose" in v and "4.00x" in v for v in c.violations)


def test_cost_increase_inside_tolerance_passes():
    base = artifact(cost=0.010, results=[case("x", True)])
    new = artifact(cost=0.011, results=[case("x", True)])
    assert compare(base, new, max_cost_increase_pct=20).ok


def test_cost_reduction_is_an_improvement_not_a_violation():
    base = artifact(cost=0.040, results=[case("x", True)])
    new = artifact(cost=0.010, results=[case("x", True)])
    c = compare(base, new)
    assert c.ok
    cost_delta = next(d for d in c.deltas if d.label == "total cost")
    assert cost_delta.improved


def test_pass_rate_drop_fails_by_default():
    base = artifact(pass_rate=1.0, results=[case("x", True)])
    new = artifact(pass_rate=0.8, results=[case("x", False)])
    c = compare(base, new)
    assert not c.ok
    assert any("pass rate dropped" in v for v in c.violations)


def test_pass_rate_drop_within_explicit_tolerance():
    base = artifact(pass_rate=1.0, results=[case("x", True)])
    new = artifact(pass_rate=0.95, results=[case("x", True)])
    assert compare(base, new, max_pass_rate_drop_pct=10).ok


def test_newly_failing_cases_are_named():
    base = artifact(results=[case("a", True), case("b", True)])
    new = artifact(results=[case("a", True), case("b", False)])
    c = compare(base, new, max_pass_rate_drop_pct=100)
    assert c.newly_failing == ["b@echo-1"]
    assert not c.ok


def test_allow_new_failures_flag():
    base = artifact(results=[case("b", True)])
    new = artifact(results=[case("b", False)])
    c = compare(base, new, max_pass_rate_drop_pct=100, allow_new_failures=True)
    assert c.ok
    assert c.newly_failing == ["b@echo-1"]


def test_newly_passing_tracked_separately():
    base = artifact(results=[case("a", False)])
    new = artifact(results=[case("a", True)])
    c = compare(base, new)
    assert c.newly_passing == ["a@echo-1"]
    assert c.ok


def test_added_and_removed_cases():
    base = artifact(results=[case("a", True), case("gone", True)])
    new = artifact(results=[case("a", True), case("fresh", True)])
    c = compare(base, new)
    assert c.added == ["fresh@echo-1"]
    assert c.removed == ["gone@echo-1"]


def test_same_case_different_model_is_a_distinct_key():
    base = artifact(results=[case("a", True, "gpt-4o-mini")])
    new = artifact(results=[case("a", True, "claude-3-5-haiku-20241022")])
    c = compare(base, new)
    assert c.added == ["a@claude-3-5-haiku-20241022"]
    assert c.removed == ["a@gpt-4o-mini"]


def test_latency_regression_detected():
    base = artifact(p95=1000.0, results=[case("x", True)])
    new = artifact(p95=2000.0, results=[case("x", True)])
    c = compare(base, new)
    assert not c.ok
    assert any("p95 latency rose" in v for v in c.violations)


def test_zero_to_nonzero_cost_is_flagged():
    base = artifact(cost=0.0, results=[case("x", True)])
    new = artifact(cost=0.02, results=[case("x", True)])
    c = compare(base, new)
    assert not c.ok
    assert any("$0" in v for v in c.violations)


def test_deltas_cover_all_tracked_metrics():
    c = compare(artifact(results=[case("x", True)]), artifact(results=[case("x", True)]))
    labels = {d.label for d in c.deltas}
    assert labels == {"pass rate", "mean score", "total cost",
                      "total tokens", "p50 latency", "p95 latency"}

def test_arrow_reflects_goodness_not_numeric_direction():
    """A rising cost must read as a down-arrow: the arrow means better/worse."""
    base = artifact(cost=0.010, pass_rate=0.75, results=[case("x", True)])
    new = artifact(cost=0.096, pass_rate=1.0, results=[case("x", True)])
    c = compare(base, new)
    by = {d.label: d for d in c.deltas}
    assert by["total cost"].arrow() == "v"      # number up, outcome bad
    assert by["pass rate"].arrow() == "^"       # number up, outcome good
    assert by["p50 latency"].arrow() == "="


def test_ratio_surfaces_multiplier_for_blowups():
    base = artifact(cost=0.00002700, results=[case("x", True)])
    new = artifact(cost=0.00025800, results=[case("x", True)])
    c = compare(base, new)
    cost = next(d for d in c.deltas if d.label == "total cost")
    assert round(cost.ratio, 1) == 9.6
    assert not c.ok
