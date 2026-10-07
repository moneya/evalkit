"""Numeric comparisons in json_path.

Quality metrics are thresholds, not equalities. A retrieval gate needs to say
"recall@10 >= 0.85"; before these operators the only way to express that was
brittle string matching on a formatted number.

The bool exclusion matters more than it looks: in Python `True >= 0.85` is True,
so without it a suite asserting a float threshold would silently pass on a
boolean field and the gate would be decorative.
"""

from __future__ import annotations

import pytest

from evalkit.assertions import run_assertions
from evalkit.providers import Completion, Usage


def completion(text: str) -> Completion:
    return Completion(
        text=text,
        model="gpt-5-mini",
        usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
        latency_ms=1,
    )


def check(spec: dict, text: str) -> bool:
    results = run_assertions([{"json_path": spec}], completion(text))
    return all(r.passed for r in results)


def reason(spec: dict, text: str) -> str:
    results = run_assertions([{"json_path": spec}], completion(text))
    return " ".join(r.detail for r in results)


METRICS = '{"recall": 0.898, "precision": 0.75, "count": 17, "ok": true, "name": "run"}'


# -- gte ------------------------------------------------------------------

def test_gte_passes_when_above():
    assert check({"path": "recall", "gte": 0.85}, METRICS)


def test_gte_passes_at_the_boundary():
    """A threshold of exactly the measured value must not fail the build."""
    assert check({"path": "recall", "gte": 0.898}, METRICS)


def test_gte_fails_when_below():
    assert not check({"path": "recall", "gte": 0.95}, METRICS)


def test_gte_failure_names_both_numbers():
    detail = reason({"path": "recall", "gte": 0.95}, METRICS)
    assert "0.898" in detail and "0.95" in detail


# -- lte, gt, lt ----------------------------------------------------------

def test_lte_passes_when_below():
    assert check({"path": "precision", "lte": 0.8}, METRICS)


def test_lte_fails_when_above():
    assert not check({"path": "recall", "lte": 0.5}, METRICS)


def test_gt_is_strict_at_the_boundary():
    assert not check({"path": "recall", "gt": 0.898}, METRICS)
    assert check({"path": "recall", "gt": 0.897}, METRICS)


def test_lt_is_strict_at_the_boundary():
    assert not check({"path": "recall", "lt": 0.898}, METRICS)
    assert check({"path": "recall", "lt": 0.899}, METRICS)


# -- integers and nesting -------------------------------------------------

def test_works_on_integers():
    assert check({"path": "count", "gte": 10}, METRICS)
    assert not check({"path": "count", "gte": 20}, METRICS)


def test_works_on_nested_paths():
    doc = '{"summary": {"retrieval": {"recall@10": 0.9}}}'
    assert check({"path": "summary.retrieval.recall@10", "gte": 0.85}, doc)


# -- the dangerous cases --------------------------------------------------

def test_booleans_are_rejected_not_coerced():
    """`True >= 0.85` is True in Python — that would make a gate decorative."""
    assert not check({"path": "ok", "gte": 0.85}, METRICS)
    assert "not a number" in reason({"path": "ok", "gte": 0.85}, METRICS)


def test_strings_are_rejected_not_coerced():
    assert not check({"path": "name", "gte": 0.5}, METRICS)
    assert "not a number" in reason({"path": "name", "gte": 0.5}, METRICS)


def test_numeric_string_is_still_rejected():
    """"0.9" is not a number; silently parsing it would hide a schema change."""
    assert not check({"path": "v", "gte": 0.5}, '{"v": "0.9"}')


def test_null_is_rejected():
    assert not check({"path": "v", "gte": 0.5}, '{"v": null}')


def test_missing_path_fails_rather_than_passing_vacuously():
    """A renamed metric must fail the gate, not skip it."""
    assert not check({"path": "nope", "gte": 0.5}, METRICS)


# -- interaction with existing operators ----------------------------------

def test_equals_still_works():
    assert check({"path": "count", "equals": 17}, METRICS)


def test_contains_still_works():
    assert check({"path": "name", "contains": "ru"}, METRICS)


def test_exists_still_works():
    assert check({"path": "recall", "exists": True}, METRICS)


def test_bare_path_still_means_presence():
    assert check({"path": "recall"}, METRICS)


@pytest.mark.parametrize("op", ["gte", "lte", "gt", "lt"])
def test_every_operator_is_wired(op):
    # 0.898 vs 0.898: gte/lte pass, gt/lt fail. Either way it must not error.
    results = run_assertions(
        [{"json_path": {"path": "recall", op: 0.898}}], completion(METRICS)
    )
    assert len(results) == 1
    assert results[0].kind == "json_path"
