# evalkit

**Prompt and agent evals where cost regressions fail the build.**

Most eval tools answer "did quality change?" That's half the question. A prompt
change that improves accuracy 2% while making inference 4x more expensive is a
regression, and it will ship silently unless something is watching the bill.

evalkit treats **cost, latency, and quality as co-equal gates**. One command in
CI, non-zero exit on regression.

```
$ evalkit run examples/support_triage.yaml --baseline runs/main.json

========================================================================
  REGRESSION CHECK
========================================================================

  pass rate            90.0% ->        92.0%  ^    +2.2%  better
  mean score           0.910 ->        0.930  ^    +2.2%  better
  total cost        $0.010400 ->    $0.041600  v  +300.0%  worse   <- 4.0x
  p95 latency          910ms ->       2100ms  v  +130.8%  worse

  RESULT: FAIL
    - cost rose 300.0% (4.00x) ($0.010400 -> $0.041600), tolerance 20%
    - p95 latency rose 130.8% (910ms -> 2100ms), tolerance 50%
========================================================================
```

That run would have been merged by a quality-only harness.

---

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

## Try it with no API key

The `echo` provider is deterministic and offline, so the harness is usable and
testable with zero credentials:

```bash
evalkit run examples/offline_demo.yaml
```

## Write a suite

```yaml
name: support-triage-v1
system: |
  Classify the integration failure. Reply with JSON only:
  {"category": "...", "severity": "low|medium|high", "escalate": true|false}

models: [claude-3-5-haiku-20241022, gpt-4o-mini]   # run both, compare
max_tokens: 300
temperature: 0

thresholds:                 # absolute gates
  min_pass_rate: 0.9
  max_total_cost: 0.05
  max_p95_latency_ms: 8000

cases:
  - id: auth-401
    tags: [auth]
    prompt: |
      Client reports 401 with {"type":"authentication_error"}
    assert:
      - json_valid: true
      - json_path: {path: category, equals: auth}
      - max_tokens: 150          # budget assertions are first-class
      - max_cost: 0.002
```

Run it:

```bash
export ANTHROPIC_API_KEY=...
evalkit run examples/support_triage.yaml --save runs/main.json
```

Then gate future changes against that baseline:

```bash
evalkit run examples/support_triage.yaml --baseline runs/main.json
```

## Assertions

| Text | Structured | Budget |
|---|---|---|
| `contains`, `not_contains` | `json_valid` | `max_tokens` |
| `contains_all`, `contains_any` | `json_path` | `max_cost` |
| `regex`, `not_regex` | `json_keys` | `max_latency_ms` |
| `equals`, `iequals`, `one_of` | | |
| `word_count` | | |

`json_valid` and `json_path` tolerate ```` ```json ```` fences and surrounding
prose, because models add them. `json_path` supports `meds[0].name` indexing.

## CLI

```bash
evalkit run SUITE [--model M ...] [--provider P] [--dry-run]
                  [--tag T] [--filter SUBSTR] [-c N]
                  [--save PATH] [--baseline PATH]
                  [--min-pass-rate 0.9] [--max-cost 0.05]

evalkit compare BASELINE CURRENT [--json]
                [--max-cost-increase 20] [--max-pass-drop 0]
                [--max-latency-increase 50] [--allow-new-failures]

evalkit providers      # providers, assertions, pricing table
```

Exit codes: `0` pass, `1` regression or gate violation, `2` bad input.

## CI

```yaml
- run: uv pip install -e .
- run: evalkit run evals/triage.yaml --baseline evals/baselines/main.json
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Design notes

- **Cost is computed from real token counts** returned by the provider, not
  estimated from string length. Pricing table in `providers.py`.
- **Provider failures become case failures**, never a crashed run. A missing API
  key fails that case with a readable message; the rest of the suite proceeds.
- **Results are order-stable** even under concurrency, so artifact diffs stay
  readable.
- **Cases are keyed `case_id@model`**, so running two models produces two
  independently gated data points.
- **No LLM-as-judge in v1.** Deterministic assertions only. A grader you can't
  trust can't gate a build.

## Tests

```bash
pytest -q
```

## Roadmap

- `llm_rubric` assertion with a pinned grader model and its own eval
- HTML report and per-case cost attribution
- Retrieval metrics (recall@k, MRR) for RAG suites
- Trace export for multi-step agent runs

MIT.
