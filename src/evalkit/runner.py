"""Suite loading and execution."""

from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

import yaml

from .assertions import run_assertions
from .providers import (
    ProviderError,
    get_provider,
    infer_provider,
    resolve_provider_name,
)
from .types import CaseResult, Completion, RunResult, Usage


@dataclass
class Case:
    id: str
    prompt: str
    system: str | None = None
    assertions: list[dict[str, Any]] = field(default_factory=list)
    vars: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    fixture: str | None = None


@dataclass
class Suite:
    name: str
    cases: list[Case]
    models: list[str]
    prompt_template: str | None = None
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    provider: str | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    path: Path | None = None
    # custom endpoint support
    base_url: str | None = None
    api_key_env: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout: float | None = None
    max_retries: int | None = None

    def connection(self) -> dict[str, Any]:
        """Transport options handed to the provider constructor."""
        return {
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "extra_headers": self.extra_headers or None,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


class SuiteError(ValueError):
    pass


def load_suite(path: str | Path) -> Suite:
    p = Path(path)
    if not p.exists():
        raise SuiteError(f"suite file not found: {p}")
    try:
        doc = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise SuiteError(f"{p}: invalid YAML: {e}") from e
    if not isinstance(doc, dict):
        raise SuiteError(f"{p}: top level must be a mapping")

    raw_cases = doc.get("cases")
    if not raw_cases:
        raise SuiteError(f"{p}: suite has no `cases`")

    models = doc.get("models") or ([doc["model"]] if doc.get("model") else ["echo-1"])
    if isinstance(models, str):
        models = [models]

    cases: list[Case] = []
    for i, rc in enumerate(raw_cases):
        if not isinstance(rc, dict):
            raise SuiteError(f"{p}: case #{i + 1} must be a mapping")
        cid = str(rc.get("id") or f"case-{i + 1}")
        if "prompt" not in rc and "vars" not in rc:
            raise SuiteError(f"{p}: case {cid!r} needs `prompt` or `vars`")

        fixture = rc.get("fixture")
        fixture_file = rc.get("fixture_file")
        if fixture is not None and fixture_file is not None:
            raise SuiteError(
                f"{p}: case {cid!r} sets both `fixture` and `fixture_file` — "
                f"pick one"
            )
        if fixture_file is not None:
            # Resolved relative to the suite, so a suite and the artefact it
            # gates can live together and move together. Reading at load time
            # means a missing or unreadable file fails the run immediately
            # rather than looking like an assertion failure later.
            fpath = (p.parent / str(fixture_file)).resolve()
            try:
                fixture = fpath.read_text(encoding="utf-8")
            except FileNotFoundError:
                raise SuiteError(
                    f"{p}: case {cid!r} fixture_file not found: {fpath}. "
                    f"Generate it before running the suite."
                ) from None
            except OSError as exc:
                raise SuiteError(
                    f"{p}: case {cid!r} cannot read fixture_file {fpath}: {exc}"
                ) from None

        cases.append(
            Case(
                id=cid,
                prompt=rc.get("prompt", ""),
                system=rc.get("system"),
                assertions=rc.get("assert") or rc.get("assertions") or [],
                vars=rc.get("vars") or {},
                tags=[str(t) for t in (rc.get("tags") or [])],
                fixture=fixture,
            )
        )

    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SuiteError(f"{p}: duplicate case ids: {sorted(dupes)}")

    headers = doc.get("extra_headers") or doc.get("headers") or {}
    if headers and not isinstance(headers, dict):
        raise SuiteError(f"{p}: `extra_headers` must be a mapping")

    return Suite(
        name=doc.get("name") or p.stem,
        cases=cases,
        models=[str(m) for m in models],
        prompt_template=doc.get("prompt"),
        system=doc.get("system"),
        max_tokens=int(doc.get("max_tokens", 1024)),
        temperature=float(doc.get("temperature", 0.0)),
        provider=doc.get("provider"),
        thresholds=doc.get("thresholds") or {},
        path=p,
        base_url=doc.get("base_url"),
        api_key_env=doc.get("api_key_env"),
        extra_headers={str(k): str(v) for k, v in headers.items()},
        timeout=float(doc["timeout"]) if doc.get("timeout") is not None else None,
        max_retries=int(doc["max_retries"]) if doc.get("max_retries") is not None else None,
    )


def render_prompt(suite: Suite, case: Case) -> str:
    """Case prompt wins; otherwise the suite template with ${var} substitution."""
    if case.prompt:
        text = case.prompt
    elif suite.prompt_template:
        text = suite.prompt_template
    else:
        raise SuiteError(f"case {case.id!r}: no prompt and suite has no `prompt` template")
    if case.vars:
        try:
            return Template(text).safe_substitute(case.vars)
        except Exception as e:
            raise SuiteError(f"case {case.id!r}: template error: {e}") from e
    return text


def _run_one(suite: Suite, case: Case, model: str) -> CaseResult:
    try:
        provider_name = suite.provider or infer_provider(model)
        provider = get_provider(provider_name, **suite.connection())
    except Exception as e:
        return CaseResult(
            case.id, model, False, "", [], Usage(), 0.0,
            error=f"{type(e).__name__}: {e}", tags=case.tags,
        )
    try:
        prompt = render_prompt(suite, case)
    except SuiteError as e:
        return CaseResult(case.id, model, False, "", [], Usage(), 0.0, error=str(e), tags=case.tags)

    kwargs: dict[str, Any] = {
        "system": case.system or suite.system,
        "model": model,
        "max_tokens": suite.max_tokens,
        "temperature": suite.temperature,
    }
    if resolve_provider_name(provider_name) == "echo":
        kwargs["fixture"] = case.fixture

    try:
        completion: Completion = provider.complete(prompt, **kwargs)
    except (ProviderError, Exception) as e:
        return CaseResult(
            case.id, model, False, "", [], Usage(), 0.0,
            error=f"{type(e).__name__}: {e}", tags=case.tags,
        )

    results = run_assertions(case.assertions, completion)
    passed = all(a.passed for a in results)
    return CaseResult(
        case_id=case.id,
        model=model,
        passed=passed,
        output=completion.text,
        assertions=results,
        usage=completion.usage,
        latency_ms=completion.latency_ms,
        tags=case.tags,
    )


def run_suite(
    suite: Suite,
    *,
    concurrency: int = 4,
    only_tags: list[str] | None = None,
    filter_id: str | None = None,
    on_result=None,
) -> RunResult:
    """Execute every case against every model, in parallel."""
    cases = suite.cases
    if only_tags:
        want = set(only_tags)
        cases = [c for c in cases if want & set(c.tags)]
    if filter_id:
        cases = [c for c in cases if filter_id in c.id]
    if not cases:
        raise SuiteError("no cases matched the given filters")

    jobs = [(c, m) for m in suite.models for c in cases]
    started = time.time()
    results: list[CaseResult] = []

    if concurrency <= 1:
        for c, m in jobs:
            r = _run_one(suite, c, m)
            results.append(r)
            if on_result:
                on_result(r)
    else:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_run_one, suite, c, m): (c, m) for c, m in jobs}
            for fut in cf.as_completed(futures):
                r = fut.result()
                results.append(r)
                if on_result:
                    on_result(r)

    order = {(c.id, m): i for i, (c, m) in enumerate(jobs)}
    results.sort(key=lambda r: order.get((r.case_id, r.model), 0))

    return RunResult(
        suite=suite.name,
        results=results,
        started_at=started,
        finished_at=time.time(),
        meta={
            "models": suite.models,
            "provider": suite.provider,
            "base_url": suite.base_url,
            "suite_path": str(suite.path) if suite.path else None,
            "case_count": len(cases),
            "thresholds": suite.thresholds,
        },
    )


def check_thresholds(run: RunResult, thresholds: dict[str, float]) -> list[str]:
    """Return human-readable violations of absolute gates."""
    v: list[str] = []
    if "min_pass_rate" in thresholds and run.pass_rate < float(thresholds["min_pass_rate"]):
        v.append(f"pass rate {run.pass_rate:.1%} < required {float(thresholds['min_pass_rate']):.1%}")
    if "min_mean_score" in thresholds and run.mean_score < float(thresholds["min_mean_score"]):
        v.append(f"mean score {run.mean_score:.3f} < required {thresholds['min_mean_score']}")
    if "max_total_cost" in thresholds and run.total_cost > float(thresholds["max_total_cost"]):
        v.append(f"total cost ${run.total_cost:.6f} > budget ${float(thresholds['max_total_cost']):.6f}")
    if "max_p95_latency_ms" in thresholds and run.latency_p(95) > float(thresholds["max_p95_latency_ms"]):
        v.append(f"p95 latency {run.latency_p(95):.0f}ms > budget {thresholds['max_p95_latency_ms']}ms")
    return v
