"""Tests for suite loading, running, cost math, and the regression gate."""

from __future__ import annotations

import json

import pytest

from evalkit.compare import compare
from evalkit.pricing import cost_of, lookup
from evalkit.providers import infer_provider
from evalkit.runner import (
    SuiteError,
    check_thresholds,
    load_suite,
    render_prompt,
    run_suite,
)

SUITE = """
name: t
provider: echo
models: [echo-1]
cases:
  - id: ok
    prompt: hello
    fixture: "hello world"
    assert:
      - contains: world
  - id: bad
    prompt: hello
    fixture: "nope"
    assert:
      - contains: world
"""


def write(tmp_path, text, name="s.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


# -- loading --------------------------------------------------------------

def test_load_and_run_mixed_results(tmp_path):
    run = run_suite(load_suite(write(tmp_path, SUITE)), concurrency=1)
    assert run.total == 2
    assert run.passed == 1
    assert run.failed == 1
    assert run.pass_rate == 0.5


def test_missing_file(tmp_path):
    with pytest.raises(SuiteError, match="not found"):
        load_suite(tmp_path / "nope.yaml")


def test_suite_with_no_cases(tmp_path):
    with pytest.raises(SuiteError, match="no `cases`"):
        load_suite(write(tmp_path, "name: x\nmodels: [echo-1]\n"))


def test_duplicate_case_ids_rejected(tmp_path):
    bad = """
name: t
provider: echo
cases:
  - {id: a, prompt: x}
  - {id: a, prompt: y}
"""
    with pytest.raises(SuiteError, match="duplicate case ids"):
        load_suite(write(tmp_path, bad))


def test_invalid_yaml(tmp_path):
    with pytest.raises(SuiteError, match="invalid YAML"):
        load_suite(write(tmp_path, "name: [unclosed\n"))


def test_case_needs_prompt_or_vars(tmp_path):
    with pytest.raises(SuiteError, match="needs `prompt` or `vars`"):
        load_suite(write(tmp_path, "name: t\ncases:\n  - {id: a}\n"))


# -- templating -----------------------------------------------------------

def test_template_substitution(tmp_path):
    s = """
name: t
provider: echo
prompt: "Report ${metric} for ${name}."
cases:
  - id: a
    vars: {metric: BP, name: Okafor}
    assert:
      - contains: "Report BP for Okafor."
"""
    run = run_suite(load_suite(write(tmp_path, s)), concurrency=1)
    assert run.passed == 1


def test_missing_var_left_intact_not_crashing(tmp_path):
    suite = load_suite(write(tmp_path, """
name: t
provider: echo
prompt: "hi ${who}"
cases:
  - {id: a, vars: {other: 1}}
"""))
    assert render_prompt(suite, suite.cases[0]) == "hi ${who}"


# -- filtering and concurrency -------------------------------------------

def test_tag_filter(tmp_path):
    s = """
name: t
provider: echo
cases:
  - {id: a, prompt: x, tags: [keep]}
  - {id: b, prompt: y, tags: [drop]}
"""
    run = run_suite(load_suite(write(tmp_path, s)), concurrency=1, only_tags=["keep"])
    assert run.total == 1
    assert run.results[0].case_id == "a"


def test_id_substring_filter(tmp_path):
    run = run_suite(load_suite(write(tmp_path, SUITE)), concurrency=1, filter_id="ok")
    assert run.total == 1


def test_filter_matching_nothing_raises(tmp_path):
    with pytest.raises(SuiteError, match="no cases matched"):
        run_suite(load_suite(write(tmp_path, SUITE)), filter_id="zzz")


def test_results_keep_suite_order_under_concurrency(tmp_path):
    cases = "\n".join(f"  - {{id: c{i}, prompt: p{i}}}" for i in range(12))
    suite = load_suite(write(tmp_path, f"name: t\nprovider: echo\ncases:\n{cases}\n"))
    run = run_suite(suite, concurrency=6)
    assert [r.case_id for r in run.results] == [f"c{i}" for i in range(12)]


def test_multi_model_runs_every_case_per_model(tmp_path):
    s = """
name: t
provider: echo
models: [echo-1, echo-2]
cases:
  - {id: a, prompt: x}
  - {id: b, prompt: y}
"""
    run = run_suite(load_suite(write(tmp_path, s)), concurrency=2)
    assert run.total == 4
    assert set(run.by_model()) == {"echo-1", "echo-2"}


# -- cost math ------------------------------------------------------------

def test_cost_exact_model():
    # claude-haiku-4-5: $1.00 in / $5.00 out per 1M
    assert cost_of("claude-haiku-4-5", 1_000_000, 0) == pytest.approx(1.00)
    assert cost_of("claude-haiku-4-5", 0, 1_000_000) == pytest.approx(5.00)


def test_cost_scales_linearly():
    a = cost_of("gpt-4o-mini", 1000, 500)
    b = cost_of("gpt-4o-mini", 2000, 1000)
    assert b == pytest.approx(a * 2)


def test_unknown_model_is_unpriced_not_guessed():
    """The old behaviour invented $1/$3 for anything unknown. Never again."""
    assert lookup("some-unreleased-model-9") is None
    assert cost_of("some-unreleased-model-9", 1_000_000, 1_000_000) is None


def test_similar_model_ids_are_not_conflated():
    """gpt-5.5-pro is 6x gpt-5.5; prefix matching would under-report it."""
    assert lookup("gpt-5.5") == (5.00, 30.00)
    assert lookup("gpt-5.5-pro") == (30.00, 180.00)


def test_provider_inference():
    assert infer_provider("claude-haiku-4-5") == "anthropic"
    assert infer_provider("gpt-5-mini") == "openai"
    assert infer_provider("echo-1") == "echo"
    with pytest.raises(Exception):
        infer_provider("llama-3")


def test_missing_api_key_is_a_case_error_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = """
name: t
models: [claude-haiku-4-5]
cases:
  - {id: a, prompt: x}
"""
    run = run_suite(load_suite(write(tmp_path, s)), concurrency=1)
    assert run.failed == 1
    assert "ANTHROPIC_API_KEY" in run.results[0].error


# -- percentiles and thresholds ------------------------------------------

def test_latency_percentiles(tmp_path):
    run = run_suite(load_suite(write(tmp_path, SUITE)), concurrency=1)
    for r, ms in zip(run.results, [100.0, 300.0]):
        r.latency_ms = ms
    assert run.latency_p(50) == pytest.approx(200.0)
    assert run.latency_p(0) == pytest.approx(100.0)
    assert run.latency_p(100) == pytest.approx(300.0)


def test_threshold_violations(tmp_path):
    run = run_suite(load_suite(write(tmp_path, SUITE)), concurrency=1)
    v = check_thresholds(run, {"min_pass_rate": 0.9})
    assert len(v) == 1
    assert "pass rate" in v[0]
    assert check_thresholds(run, {"min_pass_rate": 0.5}) == []


def test_cost_threshold_violation(tmp_path):
    run = run_suite(load_suite(write(tmp_path, SUITE)), concurrency=1)
    run.results[0].usage.cost_usd = 0.5
    v = check_thresholds(run, {"max_total_cost": 0.01})
    assert any("total cost" in x for x in v)


def test_run_artifact_roundtrips_as_json(tmp_path):
    run = run_suite(load_suite(write(tmp_path, SUITE)), concurrency=1)
    d = json.loads(json.dumps(run.to_dict()))
    assert d["summary"]["total"] == 2
    assert d["results"][0]["case_id"] == "ok"
