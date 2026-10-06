"""Tests for preflight validation.

The point of `evalkit validate` is that a suite mistake costs nothing to find.
These tests assert each class of mistake is caught, and — importantly — that
valid suites stay quiet, because a validator that cries wolf gets switched off.
"""

from __future__ import annotations

import pytest

from evalkit.runner import load_suite
from evalkit.validate import ERROR, INFO, WARNING, validate, worst_severity


def check(tmp_path, body: str):
    p = tmp_path / "s.yaml"
    p.write_text(body)
    return validate(load_suite(p))


def messages(findings, severity=None):
    return [f"{f.where}: {f.message}" for f in findings if severity in (None, f.severity)]


def errors(findings):
    return messages(findings, ERROR)


def warnings(findings):
    return messages(findings, WARNING)


# -- valid suites must stay quiet -----------------------------------------

def test_clean_suite_reports_only_info(tmp_path):
    f = check(tmp_path, """
name: clean
provider: echo
models: [echo-1]
cases:
  - id: a
    prompt: hello
    assert:
      - contains: hello
""")
    assert errors(f) == []
    assert warnings(f) == []
    assert worst_severity(f) == INFO


def test_every_shipped_example_is_valid_except_the_broken_one():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "examples"
    for path in sorted(root.glob("*.yaml")):
        f = validate(load_suite(path))
        if path.name == "broken_suite.yaml":
            assert errors(f), "broken_suite.yaml should report errors"
        else:
            assert errors(f) == [], f"{path.name} reported: {errors(f)}"


def test_alias_default_base_url_is_not_a_missing_base_url(tmp_path):
    """`provider: ollama` supplies its own URL; flagging it would be a false alarm."""
    f = check(tmp_path, """
name: a
provider: ollama
models: [llama3]
cases:
  - {id: a, prompt: hi, assert: [{contains: x}]}
""")
    assert not any("base_url" in m for m in errors(f))


# -- assertion mistakes ---------------------------------------------------

def test_unknown_assertion_suggests_the_right_one(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
cases:
  - {id: a, prompt: x, assert: [{contain: "y"}]}
""")
    assert any("unknown assertion 'contain'" in m for m in errors(f))
    assert any("did you mean 'contains'" in (fd.hint or "") for fd in f)


def test_two_checks_in_one_list_item(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
cases:
  - id: a
    prompt: x
    assert:
      - contains: "a"
        regex: "b"
""")
    assert any("exactly one check" in m for m in errors(f))


def test_invalid_regex_is_caught_statically(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
cases:
  - {id: a, prompt: x, assert: [{regex: "([unclosed"}]}
""")
    assert any("invalid regex" in m for m in errors(f))


def test_valid_regex_passes(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
cases:
  - {id: a, prompt: x, assert: [{regex: '\\d+%'}]}
""")
    assert errors(f) == []


@pytest.mark.parametrize("line,expect", [
    ('{json_path: "category"}', "expects a mapping"),
    ('{one_of: "yes"}', "expects a list"),
    ('{contains_all: []}', "empty list"),
    ('{max_tokens: "lots"}', "expects a number"),
    ('{max_cost: -1}', "cannot be negative"),
    ('{word_count: {}}', "needs `min` and/or `max`"),
    ('{json_path: {equals: a}}', "requires a `path`"),
    ('{json_path: {path: a, nope: 1}}', "unknown key"),
])
def test_assertion_shape_errors(tmp_path, line, expect):
    f = check(tmp_path, f"""
name: a
provider: echo
cases:
  - {{id: a, prompt: x, assert: [{line}]}}
""")
    assert any(expect in m for m in errors(f)), errors(f)


def test_json_path_without_comparison_is_only_a_warning(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
cases:
  - {id: a, prompt: x, assert: [{json_path: {path: category}}]}
""")
    assert errors(f) == []
    assert any("only checks presence" in m for m in warnings(f))


def test_zero_token_budget_is_flagged(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
cases:
  - {id: a, prompt: x, assert: [{max_tokens: 0}]}
""")
    assert any("no response can satisfy" in m for m in warnings(f))


# -- template variables ---------------------------------------------------

def test_missing_template_var_is_an_error(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
cases:
  - id: a
    prompt: "Report ${metric} for ${patient}."
    vars: {metric: HbA1c}
    assert: [{contains: HbA1c}]
""")
    assert any("'patient'" in m for m in errors(f))


def test_unused_var_is_one_warning_not_two(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
cases:
  - id: a
    prompt: "no placeholders"
    vars: {unused: 1}
    assert: [{contains: no}]
""")
    unused = [m for m in warnings(f) if "unused" in m]
    assert len(unused) == 1


def test_suite_level_template_is_resolved_against_case_vars(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
prompt: "Report ${metric}."
cases:
  - {id: a, vars: {metric: BP}, assert: [{contains: BP}]}
""")
    assert errors(f) == []


# -- thresholds -----------------------------------------------------------

def test_percentage_instead_of_fraction(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
thresholds: {min_pass_rate: 90}
cases:
  - {id: a, prompt: x, assert: [{contains: x}]}
""")
    assert any("fraction between 0 and 1" in m for m in errors(f))


def test_unknown_threshold_key_suggests_the_real_one(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
thresholds: {max_cost: 0.05}
cases:
  - {id: a, prompt: x, assert: [{contains: x}]}
""")
    assert any("unknown threshold" in m for m in errors(f))
    assert any("max_total_cost" in (fd.hint or "") for fd in f)


def test_cost_budget_on_unpriced_model_is_a_warning(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
models: [mystery-model-7]
thresholds: {max_total_cost: 0.05}
cases:
  - {id: a, prompt: x, assert: [{contains: x}]}
""")
    assert any("cannot be enforced" in m for m in warnings(f))


# -- provider / model -----------------------------------------------------

def test_uninferable_model_is_an_error(tmp_path):
    f = check(tmp_path, """
name: a
models: [mistral-7b-local]
cases:
  - {id: a, prompt: x, assert: [{contains: x}]}
""")
    assert any("cannot infer a provider" in m for m in errors(f))


def test_openai_compatible_without_base_url(tmp_path, monkeypatch):
    monkeypatch.delenv("EVALKIT_BASE_URL", raising=False)
    f = check(tmp_path, """
name: a
provider: openai_compatible
models: [whatever]
cases:
  - {id: a, prompt: x, assert: [{contains: x}]}
""")
    assert any("needs a base_url" in m for m in errors(f))


def test_unpriced_model_is_info_not_error(tmp_path):
    f = check(tmp_path, """
name: a
provider: openai_compatible
base_url: http://localhost:8000/v1
models: [some-local-model]
cases:
  - {id: a, prompt: x, assert: [{contains: x}]}
""")
    assert errors(f) == []
    assert any("not in the pricing table" in m for m in messages(f, INFO))


# -- misc -----------------------------------------------------------------

def test_case_without_assertions_warns(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
cases:
  - {id: a, prompt: x}
""")
    assert any("can never fail" in m for m in warnings(f))


def test_fixture_outside_echo_warns(tmp_path):
    f = check(tmp_path, """
name: a
provider: openai
models: [gpt-5-mini]
cases:
  - {id: a, prompt: x, fixture: canned, assert: [{contains: canned}]}
""")
    assert any("will be ignored" in m for m in warnings(f))


def test_call_count_is_reported(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
models: [echo-1, echo-2]
cases:
  - {id: a, prompt: x, assert: [{contains: x}]}
  - {id: b, prompt: y, assert: [{contains: y}]}
""")
    assert any("= 4 API call(s)" in m for m in messages(f))


def test_large_suite_warns_about_scale(tmp_path):
    cases = "\n".join(
        f"  - {{id: c{i}, prompt: p{i}, assert: [{{contains: p}}]}}" for i in range(250)
    )
    f = check(tmp_path, f"name: a\nprovider: echo\ncases:\n{cases}\n")
    assert any("API call(s)" in m for m in warnings(f))


def test_findings_are_sorted_errors_first(tmp_path):
    f = check(tmp_path, """
name: a
provider: echo
thresholds: {min_pass_rate: 90}
cases:
  - {id: a, prompt: x}
""")
    severities = [x.severity for x in f]
    assert severities == sorted(severities, key=lambda s: {ERROR: 0, WARNING: 1, INFO: 2}[s])
