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

  pass rate             75.0% ->       100.0%  ^   +33.3%  better
  mean score            0.875 ->        1.000  ^   +14.3%  better
  total cost        $0.000082 ->    $0.000849  v  +935.4%  worse  <- 10.4x
  total tokens             68 ->          463  v  +580.9%  worse   <- 6.8x

  newly passing (1)
    + bad-amount@gpt-5-mini

  RESULT: FAIL
    - cost rose 935.4% (10.35x) ($0.000082 -> $0.000849), tolerance 20%
========================================================================
```

Quality went **up** and the build still fails: the same answers now cost 10x.\nA quality-only harness merges this. Reproduce it offline with\n`examples/regression_v1.yaml` vs `regression_v2.yaml` — no API key needed.

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

models: [claude-haiku-4-5, gpt-5-mini]   # run both, compare
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

evalkit providers      # providers, default models, assertion list
evalkit pricing [MODEL]  # pricing table or one model
```

Exit codes: `0` pass, `1` regression or gate violation, `2` bad input.

## CI

```yaml
- run: uv pip install -e .
- run: evalkit run evals/triage.yaml --baseline evals/baselines/main.json
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Pricing

Prices live in `src/evalkit/data/pricing.json`, not in code, with a `_verified`
date and the source URL for each vendor. Override without touching the repo:

```bash
export EVALKIT_PRICING=/path/to/my-pricing.json   # replaces/merges entries
# or drop ./evalkit-pricing.json next to your suites
evalkit pricing                 # whole table + verification date
evalkit pricing gpt-5.5-pro     # one model; exit 1 if unpriced
```

**Unknown models are not priced by guesswork.** An earlier version fell back to
a flat `$1/$3` default for anything unrecognised, which meant a typo'd model id
produced a confident, fabricated cost and could satisfy a cost gate. Now:

- `cost_usd` is `None`, rendered `n/a`, distinct from `$0`
- a `max_cost` assertion **fails** with "not in the pricing table"
- the run is marked `fully_priced: false` and names the unpriced models
- `compare` refuses to certify the cost gate at all

```
$ evalkit run examples/unpriced_model.yaml
  FAIL  cost-cannot-be-verified [some-model-released-next-tuesday]   n/a
        max_cost: model is not in the pricing table, so cost cannot be verified

  unpriced models (cost excluded): some-model-released-next-tuesday
```

Lookup is **exact, never prefix-based**: `gpt-5.5-pro` costs 6x `gpt-5.5`, and a
prefix match would have reported the cheaper one and passed a gate it should
have failed.

`scripts/check_pricing_freshness.py` warns in CI once the table ages past 60
days. It deliberately does not scrape — a scraper that half-fails silently is
worse than a date check that tells a human to look.

## Design notes

- **Cost is computed from real token counts** returned by the provider, not
  estimated from string length.
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
