"""Tests for custom provider endpoints (vLLM, Ollama, OpenRouter, gateways).

Includes a real local HTTP server so the transport path is exercised end to end,
not just mocked.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from evalkit.providers import (
    OpenAICompatibleProvider,
    _brief_body,
    ProviderError,
    default_model_for,
    get_provider,
    infer_provider,
    join_url,
    resolve_provider_name,
)
from evalkit.runner import load_suite, run_suite


# -- URL joining ----------------------------------------------------------

@pytest.mark.parametrize(
    "base,expected",
    [
        ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1/", "http://localhost:8000/v1/chat/completions"),
        ("http://x/v1/chat/completions", "http://x/v1/chat/completions"),
        ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/chat/completions"),
        ("https://api.deepinfra.com/v1/openai", "https://api.deepinfra.com/v1/openai/v1/chat/completions"),
    ],
)
def test_join_url_tolerates_how_people_write_bases(base, expected):
    assert join_url(base, "/v1/chat/completions") == expected


# -- aliases and defaults -------------------------------------------------

def test_aliases_resolve_to_openai_compatible():
    for alias in ("vllm", "ollama", "lmstudio", "openrouter", "custom", "local"):
        assert resolve_provider_name(alias) == "openai_compatible"


def test_alias_supplies_a_default_base_url():
    assert get_provider("ollama").base_url == "http://localhost:11434/v1"
    assert get_provider("vllm").base_url == "http://localhost:8000/v1"
    assert get_provider("lmstudio").base_url == "http://localhost:1234/v1"


def test_alias_default_does_not_override_explicit_base_url():
    p = get_provider("ollama", base_url="http://gpu-box:9000/v1")
    assert p.base_url == "http://gpu-box:9000/v1"


def test_openrouter_alias_sets_its_key_env():
    p = get_provider("openrouter")
    assert p.api_key_env == "OPENROUTER_API_KEY"
    assert p.base_url == "https://openrouter.ai/api/v1"


def test_unknown_provider_lists_options():
    with pytest.raises(ProviderError, match="unknown provider"):
        get_provider("not-a-provider")


# -- required / optional configuration ------------------------------------

def test_openai_compatible_requires_a_base_url():
    with pytest.raises(ProviderError, match="needs a base_url"):
        OpenAICompatibleProvider()


def test_openai_compatible_needs_no_api_key():
    """vLLM, Ollama and LM Studio serve keyless; demanding a key would block them."""
    p = get_provider("openai_compatible", base_url="http://localhost:8000/v1")
    assert p._api_key() is None


def test_custom_endpoint_uses_max_tokens_not_max_completion_tokens():
    """max_completion_tokens is an OpenAI rename; self-hosted servers reject it."""
    assert OpenAICompatibleProvider.max_tokens_field == "max_tokens"
    from evalkit.providers import OpenAIProvider
    assert OpenAIProvider.max_tokens_field == "max_completion_tokens"


def test_custom_endpoint_has_no_default_model():
    assert default_model_for("openai_compatible") == ""
    assert default_model_for("vllm") == ""


def test_base_url_env_overrides(monkeypatch):
    monkeypatch.setenv("EVALKIT_BASE_URL", "http://env-host:1234/v1")
    p = get_provider("openai_compatible")
    assert p.base_url == "http://env-host:1234/v1"


def test_provider_specific_env_beats_generic(monkeypatch):
    monkeypatch.setenv("EVALKIT_BASE_URL", "http://generic/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://specific/v1")
    assert get_provider("openai").base_url == "http://specific/v1"


def test_explicit_arg_beats_every_env(monkeypatch):
    monkeypatch.setenv("EVALKIT_BASE_URL", "http://generic/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://specific/v1")
    p = get_provider("openai", base_url="http://explicit/v1")
    assert p.base_url == "http://explicit/v1"


def test_anthropic_base_url_is_overridable_for_proxies():
    p = get_provider("anthropic", base_url="http://corp-gateway/anthropic")
    assert p.url == "http://corp-gateway/anthropic/v1/messages"


def test_custom_endpoint_searches_all_pricing_sections():
    """A gateway can serve any vendor's models, so don't scope pricing to one."""
    p = get_provider("openai_compatible", base_url="http://x/v1")
    assert p.pricing_scope is None


def test_infer_provider_never_guesses_a_custom_endpoint():
    """Guessing openai_compatible would surface a confusing connection error."""
    with pytest.raises(ProviderError, match="openai_compatible"):
        infer_provider("mistral-7b-instruct")
    assert infer_provider("gpt-5-mini") == "openai"


# -- live local server ----------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    captured: dict = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).captured = {
            "path": self.path,
            "body": body,
            "headers": {k.lower(): v for k, v in self.headers.items()},
        }
        payload = {
            "choices": [{"message": {"role": "assistant", "content": "served locally"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture
def local_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_live_custom_endpoint_round_trip(local_server):
    p = get_provider("openai_compatible", base_url=local_server)
    c = p.complete("ping", system="be terse", model="local-llama-3-8b", max_tokens=64)

    assert c.text == "served locally"
    assert c.usage.input_tokens == 11
    assert c.usage.output_tokens == 7
    assert c.latency_ms > 0
    # unknown self-hosted model -> unpriced, not a fabricated cost
    assert c.usage.cost_usd is None

    sent = _Handler.captured
    assert sent["path"] == "/v1/chat/completions"
    assert sent["body"]["model"] == "local-llama-3-8b"
    assert sent["body"]["max_tokens"] == 64
    assert "max_completion_tokens" not in sent["body"]
    assert sent["body"]["messages"][0] == {"role": "system", "content": "be terse"}
    assert "authorization" not in sent["headers"]


def test_live_custom_endpoint_sends_key_and_extra_headers(local_server, monkeypatch):
    monkeypatch.setenv("MY_GATEWAY_KEY", "secret-value")
    p = get_provider(
        "openai_compatible",
        base_url=local_server,
        api_key_env="MY_GATEWAY_KEY",
        extra_headers={"HTTP-Referer": "https://example.test", "X-Title": "evalkit"},
    )
    p.complete("ping", model="m")
    h = _Handler.captured["headers"]
    assert h["authorization"] == "Bearer secret-value"
    assert h["http-referer"] == "https://example.test"
    assert h["x-title"] == "evalkit"


def test_suite_can_declare_a_custom_endpoint(tmp_path, local_server):
    s = tmp_path / "s.yaml"
    s.write_text(f"""
name: local
provider: openai_compatible
base_url: {local_server}
models: [local-llama-3-8b]
extra_headers:
  X-Trace: abc123
cases:
  - id: a
    prompt: hello
    assert:
      - contains: "served locally"
""")
    run = run_suite(load_suite(s), concurrency=1)
    assert run.passed == 1
    assert run.results[0].usage.cost_usd is None
    assert run.unpriced == ["local-llama-3-8b"]
    assert _Handler.captured["headers"]["x-trace"] == "abc123"


def test_suite_base_url_appears_in_run_metadata(tmp_path, local_server):
    s = tmp_path / "s.yaml"
    s.write_text(f"""
name: local
provider: vllm
base_url: {local_server}
models: [m]
cases:
  - {{id: a, prompt: x}}
""")
    run = run_suite(load_suite(s), concurrency=1)
    assert run.meta["base_url"] == local_server


def test_unreachable_endpoint_is_a_case_failure_not_a_crash(tmp_path):
    s = tmp_path / "s.yaml"
    s.write_text("""
name: dead
provider: openai_compatible
base_url: http://127.0.0.1:1/v1
max_retries: 1
models: [m]
cases:
  - {id: a, prompt: x}
""")
    run = run_suite(load_suite(s), concurrency=1)
    assert run.failed == 1
    assert "127.0.0.1:1" in run.results[0].error


def test_missing_base_url_is_a_case_failure_with_guidance(tmp_path):
    s = tmp_path / "s.yaml"
    s.write_text("""
name: no-url
provider: openai_compatible
models: [m]
cases:
  - {id: a, prompt: x}
""")
    run = run_suite(load_suite(s), concurrency=1)
    assert run.failed == 1
    assert "needs a base_url" in run.results[0].error


def test_bad_extra_headers_type_is_rejected_at_load(tmp_path):
    from evalkit.runner import SuiteError

    s = tmp_path / "s.yaml"
    s.write_text("""
name: x
provider: openai_compatible
base_url: http://x/v1
extra_headers: "not a mapping"
cases:
  - {id: a, prompt: y}
""")
    with pytest.raises(SuiteError, match="must be a mapping"):
        load_suite(s)


# -- error body readability -----------------------------------------------

def _resp(status, *, text="", ctype=None, js=None):
    import httpx

    req = httpx.Request("POST", "http://host/v1/chat/completions")
    if js is not None:
        return httpx.Response(status, json=js, request=req)
    headers = {"content-type": ctype} if ctype else {}
    return httpx.Response(status, headers=headers, text=text, request=req)


def test_html_error_page_is_summarised_not_dumped():
    """A wrong base_url hits a web root; don't bury the cause in normalize.css."""
    html = (
        "<!DOCTYPE html><html><head><title>Not Found</title>"
        "<style>/*! normalize.css v8.0.1 */" + "x" * 5000 + "</style></head></html>"
    )
    out = _brief_body(_resp(404, text=html, ctype="text/html; charset=UTF-8"))
    assert "Not Found" in out
    assert "check base_url" in out
    assert "normalize.css" not in out
    assert len(out) < 200


def test_html_detected_without_content_type_header():
    out = _brief_body(_resp(404, text="<html><head><title>Oops</title></head></html>"))
    assert "Oops" in out and "check base_url" in out


def test_json_error_message_is_extracted():
    out = _brief_body(_resp(400, js={"error": {"message": "model 'foo' not found"}}))
    assert out == "model 'foo' not found"


def test_plain_text_error_is_whitespace_collapsed():
    assert _brief_body(_resp(500, text="  internal\n  server   error  ")) == "internal server error"


def test_empty_body_still_says_something():
    assert "502" in _brief_body(_resp(502, text=""))


def test_live_html_404_becomes_one_actionable_line():
    """A base_url pointing at a web root (the classic mistake) must not dump HTML."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class WebRoot(BaseHTTPRequestHandler):
        def do_POST(self):
            page = (
                b"<!DOCTYPE html><html><head><title>Not Found</title>"
                b"<style>/*! normalize.css v8.0.1 */" + b"x" * 4000 + b"</style></head></html>"
            )
            self.send_response(404)
            self.send_header("content-type", "text/html; charset=UTF-8")
            self.send_header("content-length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), WebRoot)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        p = get_provider(
            "openai_compatible",
            base_url=f"http://127.0.0.1:{srv.server_port}/v1",
            max_retries=1,
        )
        with pytest.raises(ProviderError) as ei:
            p.complete("x", model="m")
        msg = str(ei.value)
        assert "404" in msg
        assert "Not Found" in msg
        assert "check base_url" in msg
        assert "normalize.css" not in msg
        assert len(msg) < 400
    finally:
        srv.shutdown()
