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

Quality went **up** and the build still fails: the same answers now cost 10x.
A quality-only harness merges this. Reproduce it offline with
`examples/regression_v1.yaml` vs `regression_v2.yaml` — no API key needed.

---

## Installation

**Requirements:** Python ≥ 3.10 (verified on 3.10 and 3.14). Two dependencies,
`pyyaml` and `httpx`. No API key needed to install, run the test suite, or try
the offline demo.

### Option 1 — just use the CLI (recommended)

Installs `evalkit` on your PATH in an isolated environment, without touching
your project's packages:

```bash
# with uv (https://docs.astral.sh/uv/)
uv tool install git+https://github.com/moneya/evalkit.git

# or with pipx
pipx install git+https://github.com/moneya/evalkit.git

evalkit --version          # evalkit 0.1.0
```

### Option 2 — add it to a project

```bash
# uv
uv add git+https://github.com/moneya/evalkit.git

# pip, into an existing virtualenv
pip install git+https://github.com/moneya/evalkit.git
```

Pin a commit for reproducible CI — evals that silently change behaviour defeat
the point:

```bash
uv add "git+https://github.com/moneya/evalkit.git@40ac1da"
```

### Option 3 — clone it, to hack on it or run the examples

The example suites and the stub server live in the repo, so clone if you want
them:

```bash
git clone https://github.com/moneya/evalkit.git
cd evalkit

# with uv — creates .venv and installs dev extras
uv venv
uv pip install -e ".[dev]"

# or with stdlib venv + pip
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Verify the install

```bash
# 1. is the CLI wired up?
evalkit --version                  # evalkit 0.1.0

# 2. is the bundled pricing data present?
evalkit pricing claude-haiku-4-5   # prints rates + a verification date

# 3. does a real eval run? (clone only — needs examples/)
evalkit run examples/offline_demo.yaml      # 5/5 passed

# 4. the test suite (clone + dev extras only)
pytest -q                                   # 117 passed
```

If you installed with `uv venv` and did **not** activate it, prefix the commands
with the venv path — `.venv/bin/evalkit`, `.venv/bin/pytest` — or run
`source .venv/bin/activate` first.

Expected output from step 3:

```
  running offline-demo  (5 cases x 1 model(s))

  PASS  json-shape [echo-1]        0ms   $0   14tok
  PASS  fenced-json [echo-1]       0ms   $0   22tok
  PASS  text-checks [echo-1]       0ms   $0   15tok
  PASS  template-vars [echo-1]     0ms   $0   10tok
  PASS  budget-checks [echo-1]     0ms   $0    1tok

  cases      5/5 passed  (100.0%)
  cost       $0.000000   (89 tokens)
```

Case lines stream in completion order, so yours may be shuffled — that's normal
under concurrency. Saved artifacts are always sorted back into suite order so
`compare` diffs stay readable. What matters is `5/5 passed`.

If that works the install is sound: it exercises suite loading, the provider
layer, all 16 assertions, and cost arithmetic.

### Add credentials when you want real models

Only needed for hosted providers; skip entirely for local models and the offline
demo.

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # for claude-* models
export OPENAI_API_KEY=sk-...            # for gpt-*, o3, o4 models
```

A missing key is **not** a crash — the affected cases fail with a readable
message and the rest of the suite still runs:

```
FAIL  auth-401 [claude-haiku-4-5]
      error: ProviderError: ANTHROPIC_API_KEY is not set — export it,
             or run with --provider echo.
```

For self-hosted models (vLLM, Ollama, LM Studio) no key is required at all —
see [Custom endpoints](#custom-endpoints-self-hosted-local-gateways).

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ERROR: File "setup.py" or "setup.cfg" not found` | Python 3.9 or an old pip. evalkit needs ≥ 3.10; upgrade pip (`pip install -U pip`) or use `uv`. |
| `evalkit: command not found` after installing | The venv isn't active (`source .venv/bin/activate`), or use `.venv/bin/evalkit` directly. For a global CLI use `uv tool install` / `pipx`. |
| `cannot infer provider for model '...'` | A self-hosted or gateway model. Set `provider: openai_compatible` plus `base_url:` in the suite, or pass `--base-url`. |
| `... is not in the pricing table` | Expected for self-hosted models. Either drop the `max_cost` assertion or add prices — see [Pricing](#pricing). |
| `404: Not Found (HTML response — check base_url)` | `base_url` points at a web root, not the API. It should end in `/v1`, e.g. `http://localhost:8000/v1`. |
| `404: model 'X' not found` | The server is reachable but hasn't got that model. For Ollama, `ollama pull X` first; for vLLM, check the `--model` it was launched with. |
| `Connection refused` on `localhost:11434` / `:8000` | The local model server isn't running. Start Ollama/vLLM, or test the plumbing with `python3 tests/fake_openai_server.py 8731`. |

### Uninstall

```bash
uv tool uninstall evalkit     # or: pipx uninstall evalkit
pip uninstall evalkit         # project installs
```

## Try it with no API key

The `echo` provider is deterministic and offline, so the harness is usable and
testable with zero credentials:

```bash
evalkit run examples/offline_demo.yaml
```

Want a real local model instead? The repo ships a stub OpenAI-compatible server
so you can exercise the full HTTP path without a GPU:

```bash
python3 tests/fake_openai_server.py 8731 &
evalkit run examples/local_model.yaml --base-url http://127.0.0.1:8731/v1
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

## Custom endpoints — self-hosted, local, gateways

Any OpenAI-shaped `/chat/completions` server works. Aliases carry a sensible
default base URL, so the common cases are one word:

```yaml
provider: vllm          # http://localhost:8000/v1
provider: ollama        # http://localhost:11434/v1
provider: lmstudio      # http://localhost:1234/v1
provider: llamacpp      # http://localhost:8080/v1
provider: openrouter    # https://openrouter.ai/api/v1   (OPENROUTER_API_KEY)
provider: together      # https://api.together.xyz/v1    (TOGETHER_API_KEY)
provider: groq          # https://api.groq.com/openai/v1 (GROQ_API_KEY)
```

Or spell it out, in the suite:

```yaml
provider: openai_compatible
base_url: http://gpu-box.internal:8000/v1
api_key_env: MY_GATEWAY_KEY      # omit entirely for keyless vLLM/Ollama
extra_headers:
  X-Title: evalkit
timeout: 120
max_retries: 3
models: [mistralai/Mistral-7B-Instruct-v0.3]
```

…or from the CLI, overriding any suite:

```bash
evalkit run suite.yaml \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --header 'X-Title: evalkit'

export EVALKIT_BASE_URL=http://localhost:8000/v1   # or ANTHROPIC_BASE_URL / OPENAI_BASE_URL
```

Precedence: CLI flag → suite field → `PROVIDER_BASE_URL` → `EVALKIT_BASE_URL` →
built-in default.

`evalkit providers` lists every provider, alias and default endpoint.

**Details that matter:**

- **No API key required.** vLLM, Ollama and LM Studio serve keyless, so demanding
  a key would lock them out. Set `api_key_env` only when your host needs one.
- **Sends `max_tokens`, not `max_completion_tokens`.** The latter is an
  OpenAI-specific rename that self-hosted servers reject.
- **`anthropic` and `openai` base URLs are overridable too**, for corporate
  proxies and recording gateways.
- **Self-hosted models are unpriced by default** — reported `n/a`, never guessed.
  To gate cost on a local model, supply your own figures:

```bash
cat > evalkit-pricing.json <<'JSON'
{"openai_compatible": {"mistralai/Mistral-7B-Instruct-v0.3": {"input": 0.07, "output": 0.07}}}
JSON
```

- **A wrong base_url is the most common mistake**, and it returns an HTML page.
  evalkit summarises it instead of dumping 4KB of CSS into your terminal:

```
ProviderError: openai_compatible at http://localhost:8000/v1/chat/completions
404: Not Found (HTML response — check base_url; it should point at the API
root, e.g. http://host:8000/v1)
```

See `examples/local_model.yaml`.

## Gating quality metrics, not just cost

`json_path` takes numeric thresholds (`gte`, `lte`, `gt`, `lt`), and a case can
read its completion from a file your pipeline produced:

```yaml
- id: recall-at-10
  prompt: "retrieval metrics"
  fixture_file: ../data/metrics.json
  assert:
    - json_path: {path: "retrieval.recall@10", gte: 0.85}
```

That is enough to gate a RAG pipeline's retrieval quality the same way this tool
gates cost. A worked example lives in
[fhir-rag](https://github.com/moneya/fhir-rag/blob/main/evals/retrieval_gate.yaml),
where a simulated regression produces:

```
FAIL  recall-at-10        json_path: retrieval.recall@10=0.61, want >= 0.85
FAIL  sparse-answer-queries  json_path: per_query.prediabetes=0, want >= 0.9
exit code 1
```

Two details that matter:

- **Booleans are rejected, not coerced.** In Python `True >= 0.85` is True, so a
  suite asserting a float threshold against a boolean field would silently pass
  and the gate would be decorative. Same for numeric strings: `"0.9"` fails
  rather than being parsed, because a field changing type is a schema change
  worth noticing.
- **A missing `fixture_file` fails the run, not an assertion.** "recall@10 is
  below threshold" and "nobody generated the metrics" need different fixes, so
  they produce different errors.

## Credentials

Put keys in a `.env` file, not in a shell `export` that dies with the window:

```bash
cp .env.example .env
$EDITOR .env          # fill in only what you use
```

```bash
# .env
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

evalkit loads the nearest `.env` at or above the working directory, so it works
from a subdirectory the way `git` does. `.env` is gitignored; `.env.example`
documents the variable names and is committed.

**The real environment always wins.** A value already set in the environment is
never overwritten by the file, so both of these still take precedence:

```bash
DEEPSEEK_API_KEY=sk-other evalkit run evals/triage.yaml   # inline wins
# and in CI, ${{ secrets.DEEPSEEK_API_KEY }} beats any checked-out .env
```

| Flag / variable | Effect |
|---|---|
| `--env-file PATH` | load a specific file instead of searching |
| `--no-env-file` | ignore `.env` entirely, use only the real environment |
| `EVALKIT_DEBUG=1` | print which variables were loaded — **names only, never values** |

No key is needed for the offline `echo` provider, or for self-hosted servers
like Ollama and vLLM.

## Validate before you spend

A run against a hosted model costs money and minutes. Most suite mistakes are
boring and detectable statically, so `evalkit validate` finds them for free:

```bash
evalkit validate evals/*.yaml
```

```
  examples/broken_suite.yaml  (deliberately-broken)
  ERROR   case 'typo-in-assertion' assertion #1: unknown assertion 'contain'
          hint: did you mean 'contains'?
  ERROR   case 'missing-template-var': template variable(s) ['patient'] have no value
          hint: add them under `vars:` (provided: ['metric'])
  ERROR   case 'bad-regex' assertion #1: invalid regex: unterminated character set
  ERROR   thresholds.min_pass_rate: must be a fraction between 0 and 1, got 90
          hint: 90% is 0.9, not 90
  ERROR   thresholds.max_cost: unknown threshold — it will be silently ignored
          hint: did you mean 'max_total_cost'?
  WARNING case 'no-assertions': no assertions — this case can never fail
  WARNING case 'json-path-no-comparison' assertion #1: json_path has no
          equals/contains/exists, so it only checks presence
  INFO    suite: 9 case(s) x 1 model(s) = 9 API call(s)
```

What it catches:

| Class | Examples |
|---|---|
| Typos | unknown assertion or threshold name, with a "did you mean" suggestion |
| Wrong shapes | `json_path` given a string, `one_of` given a scalar, `max_tokens` given text |
| Broken regex | compiled at validation time, not on the first API response |
| Template holes | `${patient}` with no matching `vars` entry; unused vars |
| Silent no-ops | a case with no assertions, `json_path` with nothing to compare, `max_tokens: 0` |
| Config mistakes | `min_pass_rate: 90` instead of `0.9`, a cost budget on an unpriced model |
| Dead config | `fixture` set while the provider isn't `echo`, so it will be ignored |
| Scale | total API calls the run will make, warning above 200 |

**`run` validates first by default.** A suite with errors exits `2` and sends
nothing to any model:

```
$ evalkit run examples/broken_suite.yaml
  suite has errors — nothing was sent to a model:

  ERROR   case 'bad-regex' assertion #1: invalid regex: unterminated character set
  ...
  fix these, or run `evalkit validate examples/broken_suite.yaml` for the full report.
```

Use `--no-validate` to skip the check, and `validate --strict` in CI to fail on
warnings too. `examples/broken_suite.yaml` collects one of nearly every mistake
if you want to see the output.

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
                  [--base-url URL] [--api-key-env VAR] [--header 'K: V']
                  [--timeout S] [--max-retries N]
                  [--tag T] [--filter SUBSTR] [-c N] [--no-validate]
                  [--save PATH] [--baseline PATH]
                  [--min-pass-rate 0.9] [--max-cost 0.05]

evalkit compare BASELINE CURRENT [--json]
                [--max-cost-increase 20] [--max-pass-drop 0]
                [--max-latency-increase 50] [--allow-new-failures]

evalkit validate SUITE... [--strict]   # static checks, no API calls

evalkit providers        # providers, default models, assertion list
evalkit pricing [MODEL]  # pricing table or one model
```

Exit codes: `0` pass, `1` regression or gate violation, `2` bad input.

## CI

```yaml
- run: uv pip install -e .

# fast, free, no credentials — catches typos before the paid step runs
- run: evalkit validate evals/*.yaml --strict

- run: evalkit run evals/triage.yaml --baseline evals/baselines/main.json
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Putting `validate --strict` before the paid step means a pull request with a
typo'd assertion fails in seconds instead of after spending the eval budget.

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
- **Keys come from `.env`, and the environment outranks it.** A gitignored file
  beats a shell export that vanishes; CI secrets must still beat a local file,
  or a stale checkout would silently run against the wrong account.
- **Validation runs before any request.** A typo should cost zero dollars to
  find, so `run` preflights the suite and refuses to start if it has errors.
- **No LLM-as-judge in v1.** Deterministic assertions only. A grader you can't
  trust can't gate a build.
- **`infer_provider` never guesses a custom endpoint.** Guessing
  `openai_compatible` for an unknown model id would surface a confusing
  connection error instead of "set `provider:` and `base_url:`".

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
