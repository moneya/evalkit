"""Tests for assertion checks. No network, no API keys."""

from __future__ import annotations

import pytest

from evalkit.assertions import available, run_assertions
from evalkit.types import Completion, Usage


def comp(text: str, *, otok: int = 10, cost: float = 0.0, latency: float = 100.0) -> Completion:
    return Completion(
        text=text,
        usage=Usage(input_tokens=5, output_tokens=otok, cost_usd=cost),
        latency_ms=latency,
        model="echo-1",
    )


def run(specs, c):
    return run_assertions(specs, c)


def all_passed(specs, c) -> bool:
    return all(a.passed for a in run(specs, c))


# -- text -----------------------------------------------------------------

def test_contains_case_insensitive_by_default():
    assert all_passed([{"contains": "hello"}], comp("Hello world"))


def test_contains_case_sensitive_opt_in():
    r = run([{"contains": {"text": "hello", "case_sensitive": True}}], comp("Hello world"))
    assert not r[0].passed


def test_not_contains_catches_refusal():
    r = run([{"not_contains": "I cannot"}], comp("I cannot help with that"))
    assert not r[0].passed
    assert "banned" in r[0].detail


def test_contains_all_reports_partial_score():
    r = run([{"contains_all": ["alpha", "beta", "gamma", "delta"]}], comp("alpha beta only"))
    assert not r[0].passed
    assert r[0].detail == "missing ['gamma', 'delta']"
    assert r[0].score == 0.5


def test_contains_any():
    assert all_passed([{"contains_any": ["zebra", "world"]}], comp("hello world"))


def test_regex_and_not_regex():
    assert all_passed([{"regex": r"\d+%"}], comp("up 42% today"))
    r = run([{"not_regex": {"pattern": "^Sure"}}], comp("Sure, here you go"))
    assert not r[0].passed


def test_equals_is_strict_and_iequals_is_not():
    assert all_passed([{"equals": "yes"}], comp("  yes  "))
    assert not run([{"equals": "yes"}], comp("Yes"))[0].passed
    assert all_passed([{"iequals": "yes"}], comp("YES"))


def test_one_of():
    assert all_passed([{"one_of": ["yes", "no"]}], comp("No"))
    assert not run([{"one_of": ["yes", "no"]}], comp("maybe"))[0].passed


def test_word_count_bounds():
    assert all_passed([{"word_count": {"min": 2, "max": 5}}], comp("three words here"))
    assert not run([{"word_count": {"max": 2}}], comp("three words here"))[0].passed


# -- JSON -----------------------------------------------------------------

def test_json_valid_plain():
    assert all_passed([{"json_valid": True}], comp('{"a": 1}'))


def test_json_valid_inside_markdown_fence():
    assert all_passed([{"json_valid": True}], comp('```json\n{"a": 1}\n```'))


def test_json_valid_with_surrounding_prose():
    assert all_passed([{"json_valid": True}], comp('Here you go: {"a": 1} hope that helps'))


def test_json_valid_fails_on_garbage():
    r = run([{"json_valid": True}], comp("not json at all"))
    assert not r[0].passed


def test_json_path_equals_and_nested_index():
    c = comp('{"meds": [{"name": "metformin"}], "n": 2}')
    assert all_passed([{"json_path": {"path": "meds[0].name", "equals": "metformin"}}], c)
    assert all_passed([{"json_path": {"path": "n", "equals": 2}}], c)


def test_json_path_missing_key_fails_gracefully():
    r = run([{"json_path": {"path": "nope", "equals": 1}}], comp('{"a": 1}'))
    assert not r[0].passed
    assert "KeyError" in r[0].detail


def test_json_path_contains():
    c = comp('{"summary": "patient has Type 2 diabetes"}')
    assert all_passed([{"json_path": {"path": "summary", "contains": "diabetes"}}], c)


def test_json_keys_missing():
    r = run([{"json_keys": ["a", "b"]}], comp('{"a": 1}'))
    assert not r[0].passed
    assert "'b'" in r[0].detail or "b" in r[0].detail


# -- budgets --------------------------------------------------------------

def test_max_tokens():
    assert all_passed([{"max_tokens": 20}], comp("hi", otok=10))
    assert not run([{"max_tokens": 5}], comp("hi", otok=10))[0].passed


def test_max_cost():
    assert all_passed([{"max_cost": 0.01}], comp("hi", cost=0.005))
    r = run([{"max_cost": 0.001}], comp("hi", cost=0.005))
    assert not r[0].passed
    assert "0.005" in r[0].detail


def test_max_latency():
    assert all_passed([{"max_latency_ms": 500}], comp("hi", latency=100))
    assert not run([{"max_latency_ms": 50}], comp("hi", latency=100))[0].passed


# -- robustness -----------------------------------------------------------

def test_unknown_assertion_fails_loudly_without_crashing():
    r = run([{"not_a_real_check": 1}], comp("x"))
    assert not r[0].passed
    assert "unknown assertion" in r[0].detail


def test_malformed_assertion_spec():
    r = run([{"contains": "a", "regex": "b"}], comp("x"))
    assert not r[0].passed
    assert "malformed" in r[0].detail


def test_registry_is_populated():
    kinds = available()
    assert len(kinds) == 16
    for required in ("contains", "not_contains", "regex", "json_valid",
                     "json_path", "max_cost", "max_tokens", "max_latency_ms"):
        assert required in kinds


def test_empty_assertions_is_vacuously_true():
    assert run([], comp("anything")) == []
