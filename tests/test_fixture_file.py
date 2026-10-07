"""Tests for `fixture_file`.

The point of this feature is gating an artefact a pipeline produced — retrieval
metrics, a cost report — without pasting the numbers into YAML, where they go
stale silently and a reviewer cannot tell whether they were measured or typed.

The failure mode that matters: a missing file must fail the RUN, not look like an
assertion failure. "recall@10 missing" and "nobody generated the metrics" need
different fixes, so they get different errors.
"""

from __future__ import annotations

import json

import pytest

from evalkit.runner import SuiteError, load_suite


def write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


SUITE = """
name: gate
provider: echo
models: [gpt-5-mini]
cases:
  - id: metrics
    prompt: "metrics"
    fixture_file: {ref}
    assert:
      - json_valid: true
      - json_path: {{path: "retrieval.recall@10", gte: 0.85}}
"""


def test_fixture_file_is_loaded_as_the_completion(tmp_path):
    write(tmp_path, "metrics.json", json.dumps({"retrieval": {"recall@10": 0.9}}))
    suite_path = write(tmp_path, "gate.yaml", SUITE.format(ref="metrics.json"))

    suite = load_suite(suite_path)
    assert suite.cases[0].fixture is not None
    assert json.loads(suite.cases[0].fixture)["retrieval"]["recall@10"] == 0.9


def test_path_is_relative_to_the_suite_not_the_cwd(tmp_path):
    """A suite and the artefact it gates should move together."""
    (tmp_path / "evals").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "m.json").write_text('{"retrieval": {"recall@10": 0.9}}')
    suite_path = tmp_path / "evals" / "gate.yaml"
    suite_path.write_text(SUITE.format(ref="../data/m.json"), encoding="utf-8")

    suite = load_suite(suite_path)
    assert json.loads(suite.cases[0].fixture)["retrieval"]["recall@10"] == 0.9


def test_missing_file_fails_the_run_with_an_actionable_message(tmp_path):
    """Not an assertion failure: the metrics were never generated."""
    suite_path = write(tmp_path, "gate.yaml", SUITE.format(ref="nope.json"))
    with pytest.raises(SuiteError) as exc:
        load_suite(suite_path)
    message = str(exc.value)
    assert "fixture_file not found" in message
    assert "nope.json" in message
    assert "Generate it" in message


def test_directory_instead_of_file_is_an_error(tmp_path):
    (tmp_path / "adir").mkdir()
    suite_path = write(tmp_path, "gate.yaml", SUITE.format(ref="adir"))
    with pytest.raises(SuiteError):
        load_suite(suite_path)


def test_fixture_and_fixture_file_together_is_rejected(tmp_path):
    """Two sources of truth for the same completion — pick one."""
    write(tmp_path, "m.json", "{}")
    suite_path = write(tmp_path, "gate.yaml", """
name: gate
provider: echo
models: [gpt-5-mini]
cases:
  - id: both
    prompt: "x"
    fixture: '{"a": 1}'
    fixture_file: m.json
    assert:
      - json_valid: true
""")
    with pytest.raises(SuiteError, match="both `fixture` and `fixture_file`"):
        load_suite(suite_path)


def test_inline_fixture_still_works(tmp_path):
    suite_path = write(tmp_path, "gate.yaml", """
name: gate
provider: echo
models: [gpt-5-mini]
cases:
  - id: inline
    prompt: "x"
    fixture: '{"a": 1}'
    assert:
      - json_valid: true
""")
    suite = load_suite(suite_path)
    assert suite.cases[0].fixture == '{"a": 1}'


def test_case_without_either_has_no_fixture(tmp_path):
    suite_path = write(tmp_path, "gate.yaml", """
name: gate
provider: echo
models: [gpt-5-mini]
cases:
  - id: plain
    prompt: "x"
    assert:
      - contains: x
""")
    assert load_suite(suite_path).cases[0].fixture is None


def test_gate_passes_and_fails_on_the_same_suite(tmp_path):
    """End to end: the threshold is what decides, not the plumbing."""
    from evalkit.assertions import run_assertions
    from evalkit.providers import Completion, Usage

    suite_path = write(tmp_path, "gate.yaml", SUITE.format(ref="m.json"))

    for recall, expected in ((0.9, True), (0.61, False)):
        write(tmp_path, "m.json", json.dumps({"retrieval": {"recall@10": recall}}))
        suite = load_suite(suite_path)
        completion = Completion(
            text=suite.cases[0].fixture,
            model="gpt-5-mini",
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
            latency_ms=0,
        )
        results = run_assertions(suite.cases[0].assertions, completion)
        assert all(r.passed for r in results) is expected
