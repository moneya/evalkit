"""Assertions: declarative checks run against a model's output.

Each assertion is a dict in YAML, e.g.

    assert:
      - contains: "HbA1c"
      - not_contains: "I cannot"
      - regex: "\\d+(\\.\\d+)?\\s*%"
      - json_valid: true
      - json_path: {path: "meds[0].name", equals: "metformin"}
      - equals: "yes"
      - one_of: ["yes", "no"]
      - max_tokens: 200
      - max_cost: 0.01
      - max_latency_ms: 3000
      - word_count: {max: 80}
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .types import AssertionResult, Completion

# registry: kind -> checker(spec, completion) -> AssertionResult
_CHECKS: dict[str, Callable[[Any, Completion], AssertionResult]] = {}


def check(kind: str):
    def deco(fn: Callable[[Any, Completion], AssertionResult]):
        _CHECKS[kind] = fn
        return fn

    return deco


def available() -> list[str]:
    return sorted(_CHECKS)


def _ok(kind: str, passed: bool, detail: str = "", score: float | None = None) -> AssertionResult:
    return AssertionResult(kind=kind, passed=passed, detail=detail, score=score)


def _norm(s: str, case_sensitive: bool) -> str:
    return s if case_sensitive else s.lower()


# -- text checks ----------------------------------------------------------


@check("contains")
def _contains(spec: Any, c: Completion) -> AssertionResult:
    """contains: "foo"  |  contains: {text: "foo", case_sensitive: true}"""
    cs = False
    if isinstance(spec, dict):
        needle, cs = str(spec.get("text", "")), bool(spec.get("case_sensitive", False))
    else:
        needle = str(spec)
    hit = _norm(needle, cs) in _norm(c.text, cs)
    return _ok("contains", hit, "" if hit else f"missing {needle!r}")


@check("not_contains")
def _not_contains(spec: Any, c: Completion) -> AssertionResult:
    cs = False
    if isinstance(spec, dict):
        needle, cs = str(spec.get("text", "")), bool(spec.get("case_sensitive", False))
    else:
        needle = str(spec)
    hit = _norm(needle, cs) in _norm(c.text, cs)
    return _ok("not_contains", not hit, f"found banned {needle!r}" if hit else "")


@check("contains_all")
def _contains_all(spec: Any, c: Completion) -> AssertionResult:
    needles = [str(x) for x in spec]
    missing = [n for n in needles if n.lower() not in c.text.lower()]
    frac = (len(needles) - len(missing)) / len(needles) if needles else 1.0
    return _ok(
        "contains_all",
        not missing,
        "" if not missing else f"missing {missing}",
        score=round(frac, 4),
    )


@check("contains_any")
def _contains_any(spec: Any, c: Completion) -> AssertionResult:
    needles = [str(x) for x in spec]
    hit = any(n.lower() in c.text.lower() for n in needles)
    return _ok("contains_any", hit, "" if hit else f"none of {needles} present")


@check("regex")
def _regex(spec: Any, c: Completion) -> AssertionResult:
    pattern = spec["pattern"] if isinstance(spec, dict) else str(spec)
    flags = re.IGNORECASE if isinstance(spec, dict) and spec.get("ignore_case") else 0
    hit = re.search(pattern, c.text, flags) is not None
    return _ok("regex", hit, "" if hit else f"no match for /{pattern}/")


@check("not_regex")
def _not_regex(spec: Any, c: Completion) -> AssertionResult:
    pattern = spec["pattern"] if isinstance(spec, dict) else str(spec)
    hit = re.search(pattern, c.text) is not None
    return _ok("not_regex", not hit, f"matched banned /{pattern}/" if hit else "")


@check("equals")
def _equals(spec: Any, c: Completion) -> AssertionResult:
    want = str(spec).strip()
    got = c.text.strip()
    hit = want == got
    return _ok("equals", hit, "" if hit else f"want {want!r}, got {got[:120]!r}")


@check("iequals")
def _iequals(spec: Any, c: Completion) -> AssertionResult:
    hit = str(spec).strip().lower() == c.text.strip().lower()
    return _ok("iequals", hit, "" if hit else f"want {spec!r}, got {c.text.strip()[:120]!r}")


@check("one_of")
def _one_of(spec: Any, c: Completion) -> AssertionResult:
    opts = [str(x).strip().lower() for x in spec]
    got = c.text.strip().lower()
    hit = got in opts
    return _ok("one_of", hit, "" if hit else f"{got[:60]!r} not in {opts}")


@check("word_count")
def _word_count(spec: Any, c: Completion) -> AssertionResult:
    n = len(c.text.split())
    lo, hi = spec.get("min"), spec.get("max")
    hit = (lo is None or n >= lo) and (hi is None or n <= hi)
    return _ok("word_count", hit, "" if hit else f"{n} words, want min={lo} max={hi}")


# -- structured output ----------------------------------------------------


def _extract_json(text: str) -> Any:
    """Parse JSON, tolerating ```json fences and surrounding prose."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = t.find(opener), t.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(t[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no parseable JSON in output")


@check("json_valid")
def _json_valid(spec: Any, c: Completion) -> AssertionResult:
    try:
        _extract_json(c.text)
        return _ok("json_valid", True)
    except ValueError as e:
        return _ok("json_valid", False, str(e))


def _dig(obj: Any, path: str) -> Any:
    """Walk a dotted path with [i] indexing: 'meds[0].name'."""
    cur = obj
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            cur = cur[int(part[1:-1])]
        else:
            cur = cur[part]
    return cur


@check("json_path")
def _json_path(spec: Any, c: Completion) -> AssertionResult:
    path = spec["path"]
    try:
        doc = _extract_json(c.text)
        val = _dig(doc, path)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        return _ok("json_path", False, f"{path}: {type(e).__name__}: {e}")

    if "equals" in spec:
        hit = val == spec["equals"]
        return _ok("json_path", hit, "" if hit else f"{path}={val!r}, want {spec['equals']!r}")
    if "contains" in spec:
        hit = str(spec["contains"]).lower() in str(val).lower()
        return _ok("json_path", hit, "" if hit else f"{path}={val!r} lacks {spec['contains']!r}")
    if "exists" in spec:
        want = bool(spec["exists"])
        return _ok("json_path", want, "" if want else f"{path} exists but should not")
    return _ok("json_path", True, f"{path} present")


@check("json_keys")
def _json_keys(spec: Any, c: Completion) -> AssertionResult:
    try:
        doc = _extract_json(c.text)
    except ValueError as e:
        return _ok("json_keys", False, str(e))
    if not isinstance(doc, dict):
        return _ok("json_keys", False, f"expected object, got {type(doc).__name__}")
    want = [str(k) for k in spec]
    missing = [k for k in want if k not in doc]
    return _ok("json_keys", not missing, "" if not missing else f"missing keys {missing}")


# -- budget checks (the point of the tool) --------------------------------


@check("max_tokens")
def _max_tokens(spec: Any, c: Completion) -> AssertionResult:
    n = c.usage.output_tokens
    hit = n <= int(spec)
    return _ok("max_tokens", hit, "" if hit else f"{n} output tokens > {spec}")


@check("max_cost")
def _max_cost(spec: Any, c: Completion) -> AssertionResult:
    if c.usage.cost_usd is None:
        return _ok(
            "max_cost", False,
            f"model {c.model!r} is not in the pricing table, so cost cannot be "
            f"verified — add it or drop this assertion",
        )
    hit = c.usage.cost_usd <= float(spec)
    return _ok("max_cost", hit, "" if hit else f"${c.usage.cost_usd:.6f} > ${float(spec):.6f}")


@check("max_latency_ms")
def _max_latency(spec: Any, c: Completion) -> AssertionResult:
    hit = c.latency_ms <= float(spec)
    return _ok("max_latency_ms", hit, "" if hit else f"{c.latency_ms:.0f}ms > {spec}ms")


# -- runner ---------------------------------------------------------------


def run_assertions(specs: list[dict[str, Any]], completion: Completion) -> list[AssertionResult]:
    """Evaluate every assertion spec against one completion."""
    out: list[AssertionResult] = []
    for spec in specs or []:
        if not isinstance(spec, dict) or len(spec) != 1:
            out.append(_ok("invalid", False, f"malformed assertion: {spec!r}"))
            continue
        kind, body = next(iter(spec.items()))
        fn = _CHECKS.get(kind)
        if fn is None:
            out.append(_ok(kind, False, f"unknown assertion {kind!r}; have {available()}"))
            continue
        try:
            out.append(fn(body, completion))
        except Exception as e:  # a broken check must not kill the run
            out.append(_ok(kind, False, f"{type(e).__name__}: {e}"))
    return out
