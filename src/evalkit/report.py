"""Terminal reporting."""

from __future__ import annotations

import os
import sys

from .compare import Comparison
from .types import RunResult

_NO_COLOR = bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return s if _NO_COLOR else f"\033[{code}m{s}\033[0m"


def green(s): return _c("32", s)
def red(s): return _c("31", s)
def yellow(s): return _c("33", s)
def dim(s): return _c("2", s)
def bold(s): return _c("1", s)


PASS = "PASS"
FAIL = "FAIL"


def line_for(result) -> str:
    mark = green(PASS) if result.passed else red(FAIL)
    cost = f"${result.usage.cost_usd:.6f}" if result.usage.cost_usd else "$0"
    head = f"  {mark}  {result.case_id} {dim('[' + result.model + ']')}"
    stats = dim(f"{result.latency_ms:>6.0f}ms  {cost:>10}  {result.usage.output_tokens:>4}tok")
    out = f"{head:<62} {stats}"
    if result.error:
        out += "\n" + red(f"        error: {result.error}")
    for a in result.assertions:
        if not a.passed:
            out += "\n" + red(f"        {a.kind}: {a.detail}")
    return out


def summary(run: RunResult, violations: list[str] | None = None) -> str:
    bar = "-" * 72
    rate = run.pass_rate
    color = green if rate == 1.0 else (yellow if rate >= 0.8 else red)
    lines = [
        bar,
        bold(f"  {run.suite}"),
        f"  cases      {run.passed}/{run.total} passed  " + color(f"({rate:.1%})"),
        f"  mean score {run.mean_score:.3f}",
        f"  cost       ${run.total_cost:.6f}   ({run.total_tokens:,} tokens)",
        f"  latency    p50 {run.latency_p(50):.0f}ms   p95 {run.latency_p(95):.0f}ms",
        f"  wall time  {run.finished_at - run.started_at:.1f}s",
    ]

    models = run.by_model()
    if len(models) > 1:
        lines.append("")
        lines.append(dim("  per model"))
        for m, rs in models.items():
            p = sum(1 for r in rs if r.passed)
            c = sum(r.usage.cost_usd for r in rs)
            lat = sorted(r.latency_ms for r in rs)[len(rs) // 2]
            lines.append(f"    {m:<34} {p}/{len(rs)}  ${c:.6f}  p50 {lat:.0f}ms")

    if violations:
        lines.append("")
        lines.append(red(bold("  THRESHOLD VIOLATIONS")))
        for v in violations:
            lines.append(red(f"    - {v}"))

    lines.append(bar)
    return "\n".join(lines)


def comparison_report(cmp: Comparison) -> str:
    bar = "=" * 72
    lines = [bar, bold("  REGRESSION CHECK"), bar, ""]
    for d in cmp.deltas:
        pct = "" if d.pct_change is None else f"{d.pct_change:+.1f}%"
        if d.unit == "$":
            before, after = f"${d.before:.6f}", f"${d.after:.6f}"
        elif d.unit == "%":
            before, after = f"{d.before:.1f}%", f"{d.after:.1f}%"
        elif d.unit == "ms":
            before, after = f"{d.before:.0f}ms", f"{d.after:.0f}ms"
        elif d.unit == "tok":
            before, after = f"{d.before:,.0f}", f"{d.after:,.0f}"
        else:
            before, after = f"{d.before:.3f}", f"{d.after:.3f}"

        if d.abs_change == 0:
            paint, note = dim, "same"
        elif d.improved:
            paint, note = green, "better"
        else:
            paint, note = red, "worse"
        extra = ""
        if d.ratio and d.ratio >= 1.5 and not d.improved:
            extra = red(f"  <- {d.ratio:.1f}x")
        lines.append(
            f"  {d.label:<14} {before:>12} -> {after:>12}  "
            f"{paint(f'{d.arrow()} {pct:>8}')}  {dim(note)}{extra}"
        )

    if cmp.newly_failing:
        lines += ["", red(bold(f"  newly failing ({len(cmp.newly_failing)})"))]
        lines += [red(f"    - {k}") for k in cmp.newly_failing[:12]]
    if cmp.newly_passing:
        lines += ["", green(bold(f"  newly passing ({len(cmp.newly_passing)})"))]
        lines += [green(f"    + {k}") for k in cmp.newly_passing[:12]]
    if cmp.added:
        lines += ["", dim(f"  new cases: {len(cmp.added)}")]
    if cmp.removed:
        lines += ["", yellow(f"  cases removed since baseline: {len(cmp.removed)}")]

    lines.append("")
    if cmp.ok:
        lines.append(green(bold("  RESULT: PASS — no regression")))
    else:
        lines.append(red(bold("  RESULT: FAIL")))
        for v in cmp.violations:
            lines.append(red(f"    - {v}"))
    lines.append(bar)
    return "\n".join(lines)
