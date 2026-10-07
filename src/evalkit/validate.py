"""Preflight validation — catch suite problems before spending money.

A run against a hosted model costs real money and real time. Most failures are
boring and detectable statically: a typo'd assertion name, a model nobody priced,
a `${var}` with no value, a custom provider with no base_url. `evalkit validate`
reports all of them without issuing a single API call.

Severities:
  error    the run will fail or produce meaningless results — exit 1
  warning  the run will work but something is probably not what you meant
  info     worth knowing (estimated scale, unpriced models)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from string import Template
from typing import Any

from .assertions import available as available_assertions
from .pricing import lookup
from .providers import (
    ProviderError,
    effective_base_url,
    infer_provider,
    resolve_provider_name,
)
from .runner import Suite

ERROR = "error"
WARNING = "warning"
INFO = "info"

# Assertions whose spec must be a mapping, and the keys each accepts.
_MAPPING_ASSERTIONS: dict[str, set[str]] = {
    "json_path": {
        "path", "equals", "contains", "exists",
        # Numeric thresholds, for gating quality metrics like recall@10.
        "gte", "lte", "gt", "lt",
    },
    "word_count": {"min", "max"},
}
_LIST_ASSERTIONS = {"contains_all", "contains_any", "one_of", "json_keys"}
_NUMERIC_ASSERTIONS = {"max_tokens", "max_cost", "max_latency_ms"}

_KNOWN_THRESHOLDS = {
    "min_pass_rate",
    "min_mean_score",
    "max_total_cost",
    "max_p95_latency_ms",
}


@dataclass
class Finding:
    severity: str
    where: str
    message: str
    hint: str = ""

    def __str__(self) -> str:
        base = f"{self.severity.upper():<7} {self.where}: {self.message}"
        return f"{base}\n{' ' * 8}hint: {self.hint}" if self.hint else base


def _suggest(name: str, options: list[str]) -> str:
    """Closest option by simple edit distance, for typo hints."""
    import difflib

    match = difflib.get_close_matches(name, options, n=1, cutoff=0.6)
    return match[0] if match else ""


def validate(suite: Suite) -> list[Finding]:
    """Static checks over a loaded suite. No network, no cost."""
    out: list[Finding] = []
    out += _check_provider_and_models(suite)
    out += _check_cases(suite)
    out += _check_thresholds(suite)
    out += _check_scale(suite)
    order = {ERROR: 0, WARNING: 1, INFO: 2}
    return sorted(out, key=lambda f: (order[f.severity], f.where))


def _check_provider_and_models(suite: Suite) -> list[Finding]:
    out: list[Finding] = []

    if not suite.models:
        out.append(Finding(ERROR, "suite", "no models declared",
                           "add `models: [claude-haiku-4-5]`"))
        return out

    declared = suite.provider
    for model in suite.models:
        where = f"model {model!r}"

        if declared:
            canonical = resolve_provider_name(declared)
            if canonical == "openai_compatible" and not effective_base_url(
                declared, suite.base_url
            ):
                out.append(Finding(
                    ERROR, where,
                    f"provider {declared!r} needs a base_url",
                    "set `base_url:` in the suite, pass --base-url, or use an alias "
                    "with a default (vllm, ollama, lmstudio, openrouter, ...)",
                ))
        else:
            try:
                canonical = infer_provider(model)
            except ProviderError:
                out.append(Finding(
                    ERROR, where,
                    "cannot infer a provider from this model id",
                    "set `provider:` explicitly (openai_compatible + base_url "
                    "for self-hosted models)",
                ))
                continue

        if lookup(model) is None:
            out.append(Finding(
                INFO, where,
                "not in the pricing table — cost will report n/a",
                "add prices to ./evalkit-pricing.json to enable cost gating",
            ))

    return out


def _check_cases(suite: Suite) -> list[Finding]:
    out: list[Finding] = []
    kinds = available_assertions()

    if not suite.cases:
        out.append(Finding(ERROR, "suite", "no cases"))
        return out

    for case in suite.cases:
        where = f"case {case.id!r}"

        # prompt resolution
        if not case.prompt and not suite.prompt_template:
            out.append(Finding(ERROR, where, "no prompt and suite has no `prompt` template"))
        else:
            text = case.prompt or suite.prompt_template or ""
            placeholders = set(re.findall(r"\$\{(\w+)\}", text))
            missing = placeholders - set(case.vars)
            if missing:
                out.append(Finding(
                    ERROR, where,
                    f"template variable(s) {sorted(missing)} have no value",
                    f"add them under `vars:` (provided: {sorted(case.vars) or 'none'})",
                ))
            unused = set(case.vars) - placeholders
            if unused:
                detail = (
                    "the prompt has no ${...} placeholders at all"
                    if not placeholders
                    else f"referenced: {sorted(placeholders)}"
                )
                out.append(Finding(
                    WARNING, where,
                    f"vars {sorted(unused)} are never used ({detail})",
                ))

        # assertions
        if not case.assertions:
            out.append(Finding(
                WARNING, where,
                "no assertions — this case can never fail",
                "add at least one check, or tag it as a smoke case deliberately",
            ))

        for i, spec in enumerate(case.assertions, 1):
            at = f"{where} assertion #{i}"
            if not isinstance(spec, dict):
                out.append(Finding(ERROR, at, f"must be a mapping, got {type(spec).__name__}"))
                continue
            if len(spec) != 1:
                out.append(Finding(
                    ERROR, at,
                    f"must hold exactly one check, found {sorted(spec)}",
                    "split them into separate `- ` list items",
                ))
                continue

            kind, body = next(iter(spec.items()))
            if kind not in kinds:
                sug = _suggest(kind, kinds)
                out.append(Finding(
                    ERROR, at, f"unknown assertion {kind!r}",
                    f"did you mean {sug!r}?" if sug else f"available: {', '.join(kinds)}",
                ))
                continue

            out += _check_assertion_shape(at, kind, body)

        # fixtures only mean something offline
        if case.fixture is not None and resolve_provider_name(suite.provider or "") != "echo":
            out.append(Finding(
                WARNING, where,
                "`fixture` is set but the provider is not `echo`, so it will be ignored",
            ))

    return out


def _check_assertion_shape(at: str, kind: str, body: Any) -> list[Finding]:
    out: list[Finding] = []

    if kind in _MAPPING_ASSERTIONS:
        if not isinstance(body, dict):
            out.append(Finding(
                ERROR, at, f"{kind} expects a mapping, got {type(body).__name__}",
                f"e.g. {kind}: {{path: category, equals: auth}}"
                if kind == "json_path" else f"e.g. {kind}: {{max: 80}}",
            ))
            return out
        allowed = _MAPPING_ASSERTIONS[kind]
        unknown = set(body) - allowed
        if unknown:
            out.append(Finding(
                ERROR, at, f"{kind} got unknown key(s) {sorted(unknown)}",
                f"allowed: {sorted(allowed)}",
            ))
        if kind == "json_path" and "path" not in body:
            out.append(Finding(ERROR, at, "json_path requires a `path`"))
        if kind == "json_path" and len(
            set(body) & {"equals", "contains", "exists", "gte", "lte", "gt", "lt"}
        ) == 0:
            out.append(Finding(
                WARNING, at,
                "json_path has no equals/contains/exists/gte/lte/gt/lt, so it "
                "only checks presence",
            ))
        if kind == "word_count" and not (set(body) & {"min", "max"}):
            out.append(Finding(ERROR, at, "word_count needs `min` and/or `max`"))

    elif kind in _LIST_ASSERTIONS:
        if not isinstance(body, (list, tuple)):
            out.append(Finding(
                ERROR, at, f"{kind} expects a list, got {type(body).__name__}",
                f"e.g. {kind}: [one, two]",
            ))
        elif not body:
            out.append(Finding(ERROR, at, f"{kind} got an empty list"))

    elif kind in _NUMERIC_ASSERTIONS:
        if isinstance(body, bool) or not isinstance(body, (int, float)):
            out.append(Finding(
                ERROR, at, f"{kind} expects a number, got {type(body).__name__}",
            ))
        elif body < 0:
            out.append(Finding(ERROR, at, f"{kind} cannot be negative"))
        elif body == 0 and kind != "max_cost":
            out.append(Finding(WARNING, at, f"{kind} is 0, which no response can satisfy"))

    elif kind in ("regex", "not_regex"):
        pattern = body.get("pattern") if isinstance(body, dict) else body
        if not isinstance(pattern, str):
            out.append(Finding(ERROR, at, f"{kind} needs a pattern string"))
        else:
            try:
                re.compile(pattern)
            except re.error as e:
                out.append(Finding(ERROR, at, f"invalid regex: {e}"))

    elif kind == "json_valid":
        if body is not True:
            out.append(Finding(
                WARNING, at, "json_valid only makes sense as `true`",
            ))

    return out


def _check_thresholds(suite: Suite) -> list[Finding]:
    out: list[Finding] = []
    for key, value in (suite.thresholds or {}).items():
        where = f"thresholds.{key}"
        if key not in _KNOWN_THRESHOLDS:
            sug = _suggest(key, sorted(_KNOWN_THRESHOLDS))
            out.append(Finding(
                ERROR, where, "unknown threshold — it will be silently ignored",
                f"did you mean {sug!r}?" if sug else f"known: {sorted(_KNOWN_THRESHOLDS)}",
            ))
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            out.append(Finding(ERROR, where, f"expects a number, got {type(value).__name__}"))
            continue
        if key in ("min_pass_rate", "min_mean_score") and not 0 <= value <= 1:
            out.append(Finding(
                ERROR, where, f"must be a fraction between 0 and 1, got {value}",
                "90% is 0.9, not 90",
            ))

    if suite.thresholds.get("max_total_cost") is not None:
        unpriced = [m for m in suite.models if lookup(m) is None]
        if unpriced:
            out.append(Finding(
                WARNING, "thresholds.max_total_cost",
                f"cost budget set but {unpriced} are unpriced, so the gate cannot be enforced",
                "add prices to ./evalkit-pricing.json",
            ))
    return out


def _check_scale(suite: Suite) -> list[Finding]:
    calls = len(suite.cases) * len(suite.models)
    note = f"{len(suite.cases)} case(s) x {len(suite.models)} model(s) = {calls} API call(s)"
    severity = WARNING if calls > 200 else INFO
    hint = "that is a lot for one run; consider --tag or --filter" if calls > 200 else ""
    return [Finding(severity, "suite", note, hint)]


def worst_severity(findings: list[Finding]) -> str | None:
    for level in (ERROR, WARNING, INFO):
        if any(f.severity == level for f in findings):
            return level
    return None
