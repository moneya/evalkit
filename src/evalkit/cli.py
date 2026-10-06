"""evalkit CLI.

    evalkit run suite.yaml [--save runs/x.json] [--baseline runs/base.json]
    evalkit compare runs/base.json runs/new.json
    evalkit providers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .compare import compare, load_run
from .providers import dump_pricing
from .report import comparison_report, line_for, summary
from .runner import SuiteError, check_thresholds, load_suite, run_suite


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        suite = load_suite(args.suite)
    except SuiteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.model:
        suite.models = args.model
    if args.provider:
        suite.provider = args.provider
    if args.base_url:
        suite.base_url = args.base_url
    if args.api_key_env:
        suite.api_key_env = args.api_key_env
    if args.header:
        for h in args.header:
            if ":" not in h:
                print(f"error: --header expects 'Name: value', got {h!r}", file=sys.stderr)
                return 2
            k, v = h.split(":", 1)
            suite.extra_headers[k.strip()] = v.strip()
    if args.timeout:
        suite.timeout = args.timeout
    if args.max_retries:
        suite.max_retries = args.max_retries
    if args.dry_run:
        suite.provider = "echo"
        suite.models = ["echo-1"]
        suite.base_url = None

    where = f" via {suite.base_url}" if suite.base_url else ""
    print(
        f"\n  running {suite.name}  "
        f"({len(suite.cases)} cases x {len(suite.models)} model(s)){where}\n"
    )

    try:
        run = run_suite(
            suite,
            concurrency=args.concurrency,
            only_tags=args.tag,
            filter_id=args.filter,
            on_result=None if args.quiet else (lambda r: print(line_for(r), flush=True)),
        )
    except SuiteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    thresholds = dict(suite.thresholds)
    if args.min_pass_rate is not None:
        thresholds["min_pass_rate"] = args.min_pass_rate
    if args.max_cost is not None:
        thresholds["max_total_cost"] = args.max_cost
    violations = check_thresholds(run, thresholds)

    print()
    print(summary(run, violations))

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(run.to_dict(), indent=2))
        print(f"\n  saved -> {out}")

    exit_code = 0
    if violations:
        exit_code = 1
    if run.failed and not args.no_fail_on_case:
        exit_code = 1

    if args.baseline:
        try:
            base = load_run(args.baseline)
        except FileNotFoundError as e:
            print(f"\n  warning: {e} — skipping regression check", file=sys.stderr)
            return exit_code
        cmp = compare(
            base,
            run.to_dict(),
            max_cost_increase_pct=args.max_cost_increase,
            max_pass_rate_drop_pct=args.max_pass_drop,
            max_p95_increase_pct=args.max_latency_increase,
            allow_new_failures=args.allow_new_failures,
        )
        print()
        print(comparison_report(cmp))
        if not cmp.ok:
            exit_code = 1

    return exit_code


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        base, cur = load_run(args.baseline), load_run(args.current)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    cmp = compare(
        base,
        cur,
        max_cost_increase_pct=args.max_cost_increase,
        max_pass_rate_drop_pct=args.max_pass_drop,
        max_p95_increase_pct=args.max_latency_increase,
        allow_new_failures=args.allow_new_failures,
    )
    print(comparison_report(cmp))
    if args.json:
        print(json.dumps({
            "ok": cmp.ok,
            "violations": cmp.violations,
            "newly_failing": cmp.newly_failing,
            "newly_passing": cmp.newly_passing,
            "deltas": [
                {"label": d.label, "before": d.before, "after": d.after,
                 "pct_change": d.pct_change, "improved": d.improved}
                for d in cmp.deltas
            ],
        }, indent=2))
    return 0 if cmp.ok else 1


def _cmd_providers(args: argparse.Namespace) -> int:
    from .assertions import available
    from .providers import default_model_for

    from .providers import known_aliases, known_providers

    print("providers")
    for name in known_providers():
        dm = default_model_for(name) or "(must be specified)"
        print(f"  {name:<20} default model: {dm}")

    print("\naliases (all map to openai_compatible, most with a default base_url)")
    alias_hints = {
        "vllm": "http://localhost:8000/v1",
        "ollama": "http://localhost:11434/v1",
        "lmstudio": "http://localhost:1234/v1",
        "lm_studio": "http://localhost:1234/v1",
        "llamacpp": "http://localhost:8080/v1",
        "openrouter": "https://openrouter.ai/api/v1 (OPENROUTER_API_KEY)",
        "together": "https://api.together.xyz/v1 (TOGETHER_API_KEY)",
        "groq": "https://api.groq.com/openai/v1 (GROQ_API_KEY)",
        "fireworks": "https://api.fireworks.ai/inference/v1 (FIREWORKS_API_KEY)",
        "deepinfra": "https://api.deepinfra.com/v1/openai (DEEPINFRA_API_KEY)",
    }
    for alias in sorted(known_aliases()):
        hint = alias_hints.get(alias, "base_url required")
        print(f"  {alias:<20} {hint}")
    print("\nassertions")
    for a in available():
        print(f"  - {a}")
    return 0


def _cmd_pricing(args: argparse.Namespace) -> int:
    from .pricing import known_models, lookup, verified_on

    if args.model:
        price = lookup(args.model)
        if price is None:
            print(f"{args.model}: NOT PRICED — cost gates will fail for this model.")
            print("Add it to data/pricing.json or ./evalkit-pricing.json")
            return 1
        pin, pout = price
        print(f"{args.model}: ${pin}/1M input, ${pout}/1M output "
              f"(table verified {verified_on()})")
        return 0

    print(dump_pricing())
    print(f"\n{len(known_models())} models priced.")
    return 0


def _add_cmp_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-cost-increase", type=float, default=20.0,
                   help="fail if total cost rises more than this pct (default 20)")
    p.add_argument("--max-pass-drop", type=float, default=0.0,
                   help="allowed pass-rate drop in percentage points (default 0)")
    p.add_argument("--max-latency-increase", type=float, default=50.0,
                   help="fail if p95 latency rises more than this pct (default 50)")
    p.add_argument("--allow-new-failures", action="store_true",
                   help="do not fail merely because a passing case started failing")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evalkit",
        description="Prompt and agent evals where cost regressions fail the build.",
    )
    p.add_argument("--version", action="version", version=f"evalkit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a suite")
    r.add_argument("suite", help="path to a suite YAML file")
    r.add_argument("--model", action="append", help="override models (repeatable)")
    r.add_argument("--provider",
                   help="force a provider: anthropic|openai|openai_compatible|echo, "
                        "or an alias (vllm, ollama, lmstudio, openrouter, together, groq)")
    r.add_argument("--base-url",
                   help="custom endpoint, e.g. http://localhost:8000/v1 (vLLM) "
                        "or https://openrouter.ai/api/v1")
    r.add_argument("--api-key-env",
                   help="env var holding the key for a custom endpoint "
                        "(e.g. OPENROUTER_API_KEY)")
    r.add_argument("--header", action="append", metavar="'Name: value'",
                   help="extra request header, repeatable (gateways, routing hints)")
    r.add_argument("--timeout", type=float, help="per-request timeout seconds")
    r.add_argument("--max-retries", type=int,
                   help="retry attempts per request on 429/5xx/transport errors")
    r.add_argument("--dry-run", action="store_true",
                   help="offline: use the echo provider, no API keys, no cost")
    r.add_argument("-c", "--concurrency", type=int, default=4)
    r.add_argument("--tag", action="append", help="only cases with this tag (repeatable)")
    r.add_argument("--filter", help="only cases whose id contains this substring")
    r.add_argument("--save", help="write the run artifact JSON here")
    r.add_argument("--baseline", help="compare against this saved run after executing")
    r.add_argument("-q", "--quiet", action="store_true", help="summary only")
    r.add_argument("--min-pass-rate", type=float, help="absolute gate, e.g. 0.9")
    r.add_argument("--max-cost", type=float, help="absolute total-cost budget in USD")
    r.add_argument("--no-fail-on-case", action="store_true",
                   help="exit 0 even when individual cases fail (gates still apply)")
    _add_cmp_flags(r)
    r.set_defaults(fn=_cmd_run)

    c = sub.add_parser("compare", help="diff two saved runs")
    c.add_argument("baseline")
    c.add_argument("current")
    c.add_argument("--json", action="store_true", help="also emit machine-readable JSON")
    _add_cmp_flags(c)
    c.set_defaults(fn=_cmd_compare)

    pr = sub.add_parser("providers", help="list providers and assertions")
    pr.set_defaults(fn=_cmd_providers)

    pc = sub.add_parser("pricing", help="show the pricing table, or one model's price")
    pc.add_argument("model", nargs="?", help="check a single model id")
    pc.set_defaults(fn=_cmd_pricing)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
