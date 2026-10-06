"""Model providers. Each returns a Completion with real token counts and cost.

Built in:
  anthropic           api.anthropic.com (base_url overridable)
  openai              api.openai.com (base_url overridable)
  openai_compatible   any OpenAI-shaped /chat/completions endpoint —
                      vLLM, Ollama, LM Studio, llama.cpp server, OpenRouter,
                      Together, Groq, or a corporate gateway
  echo                offline, deterministic, no network

Pricing is NOT hardcoded here; see pricing.py and data/pricing.json.
"""

from __future__ import annotations

import os
import time
from typing import Any, Protocol

import httpx

from .pricing import cost_of, known_models, lookup, render
from .types import Completion, Usage

__all__ = [
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
    "EchoProvider",
    "ProviderError",
    "get_provider",
    "infer_provider",
    "default_model_for",
    "resolve_provider_name",
    "dump_pricing",
]


class ProviderError(RuntimeError):
    pass


class Provider(Protocol):
    name: str

    def complete(
        self, prompt: str, *, system: str | None, model: str, max_tokens: int, temperature: float
    ) -> Completion: ...


def join_url(base: str, path: str) -> str:
    """Join a base URL and path, tolerating how people actually write bases.

      http://localhost:8000           -> http://localhost:8000/v1/chat/completions
      http://localhost:8000/v1        -> http://localhost:8000/v1/chat/completions
      http://localhost:8000/v1/       -> http://localhost:8000/v1/chat/completions
      .../v1/chat/completions         -> unchanged
    """
    base = base.rstrip("/")
    tail = path.strip("/")
    if not tail:
        return base
    if base.endswith("/" + tail):
        return base
    # base may already carry the leading segment of path, e.g. ".../v1" + "v1/chat/..."
    segs = tail.split("/")
    for i in range(len(segs), 0, -1):
        prefix = "/".join(segs[:i])
        if base.endswith("/" + prefix):
            rest = "/".join(segs[i:])
            return f"{base}/{rest}" if rest else base
    return f"{base}/{tail}"


def _brief_body(r: httpx.Response, limit: int = 300) -> str:
    """Readable one-line gist of an error body.

    HTML error pages (a wrong base_url usually hits one) are summarised rather
    than dumped, since a normalize.css payload buries the actual problem.
    """
    ctype = r.headers.get("content-type", "")
    text = (r.text or "").strip()
    if "html" in ctype or text[:200].lower().lstrip().startswith(("<!doctype", "<html")):
        import re as _re

        title = _re.search(r"<title[^>]*>(.*?)</title>", text, _re.S | _re.I)
        gist = title.group(1).strip() if title else "HTML error page"
        return (
            f"{gist} (HTML response — check base_url; it should point at the "
            f"API root, e.g. http://host:8000/v1)"
        )
    try:
        j = r.json()
        msg = j.get("error") if isinstance(j, dict) else None
        if isinstance(msg, dict):
            msg = msg.get("message") or msg.get("type")
        if isinstance(msg, str):
            return msg[:limit]
    except Exception:
        pass
    return " ".join(text.split())[:limit] or f"<empty {r.status_code} body>"


class _HTTPProvider:
    """Shared transport, retry, auth and base-URL resolution."""

    name = "http"
    env_key = ""
    default_base_url = ""
    path = ""
    default_model = ""
    requires_key = True
    timeout = 120.0
    max_retries = 3

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key_env: str | None = None,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_model: str | None = None,
    ) -> None:
        # precedence: explicit arg > PROVIDER_BASE_URL > EVALKIT_BASE_URL > built-in
        env_specific = os.environ.get(f"{self.name.upper()}_BASE_URL", "").strip()
        env_generic = os.environ.get("EVALKIT_BASE_URL", "").strip()
        self.base_url = (base_url or env_specific or env_generic or self.default_base_url).strip()
        if not self.base_url:
            raise ProviderError(
                f"provider {self.name!r} needs a base_url — set `base_url:` in the suite, "
                f"pass --base-url, or export {self.name.upper()}_BASE_URL"
            )
        self.api_key_env = api_key_env or self.env_key
        self._explicit_key = api_key
        self.extra_headers = dict(extra_headers or {})
        if timeout is not None:
            self.timeout = float(timeout)
        if max_retries is not None:
            self.max_retries = int(max_retries)
        if default_model:
            self.default_model = default_model

    @property
    def url(self) -> str:
        return join_url(self.base_url, self.path)

    @property
    def pricing_scope(self) -> str | None:
        """Pricing section to consult; None searches all sections."""
        return self.name

    def _api_key(self) -> str | None:
        if self._explicit_key:
            return self._explicit_key
        key = os.environ.get(self.api_key_env, "").strip() if self.api_key_env else ""
        if key:
            return key
        if self.requires_key:
            hint = self.api_key_env or f"{self.name.upper()}_API_KEY"
            raise ProviderError(
                f"{hint} is not set — export it, or run with --provider echo."
            )
        return None

    def _auth_headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"content-type": "application/json"}
        headers.update(self._auth_headers())
        headers.update(self.extra_headers)

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    r = client.post(self.url, headers=headers, json=payload)
                if r.status_code in (429, 500, 502, 503, 529):
                    raise httpx.HTTPStatusError(
                        f"{r.status_code} {r.text[:200]}", request=r.request, response=r
                    )
                if r.status_code >= 400:
                    raise ProviderError(
                        f"{self.name} {self.url} {r.status_code}: {_brief_body(r)}"
                    )
                return r.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(1.5 * (2**attempt))
        raise ProviderError(
            f"{self.name} at {self.url} failed after {self.max_retries} attempts: {last}"
        )

    def _usage(self, model: str, itok: int, otok: int) -> Usage:
        return Usage(
            input_tokens=itok,
            output_tokens=otok,
            cost_usd=cost_of(model, itok, otok, provider=self.pricing_scope),
        )


class AnthropicProvider(_HTTPProvider):
    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"
    default_base_url = "https://api.anthropic.com"
    path = "/v1/messages"
    default_model = "claude-haiku-4-5"
    api_version = "2023-06-01"

    def _auth_headers(self) -> dict[str, str]:
        h = {"anthropic-version": self.api_version}
        key = self._api_key()
        if key:
            h["x-api-key"] = key
        return h

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        model = model or self.default_model
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        t0 = time.perf_counter()
        data = self._post(payload)
        latency = (time.perf_counter() - t0) * 1000

        text = "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
        u = data.get("usage", {})
        itok, otok = int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
        return Completion(
            text=text,
            usage=self._usage(model, itok, otok),
            latency_ms=latency,
            model=model,
            raw=data,
        )


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return m.startswith(("o1", "o3", "o4")) or m.endswith("-pro")


class OpenAIProvider(_HTTPProvider):
    name = "openai"
    env_key = "OPENAI_API_KEY"
    default_base_url = "https://api.openai.com"
    path = "/v1/chat/completions"
    default_model = "gpt-5-mini"
    max_tokens_field = "max_completion_tokens"

    def _auth_headers(self) -> dict[str, str]:
        key = self._api_key()
        return {"authorization": f"Bearer {key}"} if key else {}

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        model = model or self.default_model
        if not model:
            raise ProviderError(
                f"provider {self.name!r} has no default model — name it in the suite "
                f"(`models: [...]`) or pass --model"
            )
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            self.max_tokens_field: max_tokens,
        }
        # Reasoning models reject an explicit temperature.
        if not _is_reasoning_model(model):
            payload["temperature"] = temperature

        t0 = time.perf_counter()
        data = self._post(payload)
        latency = (time.perf_counter() - t0) * 1000

        choices = data.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        u = data.get("usage", {})
        itok, otok = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
        return Completion(
            text=text,
            usage=self._usage(model, itok, otok),
            latency_ms=latency,
            model=model,
            raw=data,
        )


class OpenAICompatibleProvider(OpenAIProvider):
    """Any OpenAI-shaped /chat/completions endpoint.

    Differences from `openai`:
      - base_url is REQUIRED; there is no sensible default
      - no API key required — vLLM, Ollama and LM Studio serve keyless
      - sends `max_tokens`, which self-hosted servers accept;
        `max_completion_tokens` is an OpenAI-specific rename
      - pricing is searched across all sections, since a gateway may serve
        models from any vendor (and self-hosted models are usually unpriced,
        which evalkit reports rather than guessing)

    Suite examples:
      vLLM        {provider: openai_compatible, base_url: http://localhost:8000/v1}
      Ollama      {provider: ollama,            base_url: http://localhost:11434/v1}
      LM Studio   {provider: lmstudio,          base_url: http://localhost:1234/v1}
      OpenRouter  {provider: openrouter, base_url: https://openrouter.ai/api/v1,
                   api_key_env: OPENROUTER_API_KEY}
    """

    name = "openai_compatible"
    env_key = ""  # no conventional key; set api_key_env when the host needs one
    default_base_url = ""
    default_model = ""
    requires_key = False
    max_tokens_field = "max_tokens"

    @property
    def pricing_scope(self) -> str | None:
        return None


class EchoProvider:
    """Offline provider. Deterministic, free, no network.

    Reply resolution:
      1. `fixture` on the case (exact canned output)
      2. the prompt echoed back

    Token counts are a ~4-chars-per-token approximation, not a real tokenizer.
    Cost follows the pricing table for the given model id, so naming a priced
    model gives realistic cost arithmetic with no network call — that is how the
    regression-gate demo stays free and deterministic.
    """

    name = "echo"
    default_model = "echo-1"
    pricing_scope = None

    def __init__(self, **_: Any) -> None:
        """Accepts and ignores transport options so --base-url stays harmless."""

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        fixture: str | None = None,
    ) -> Completion:
        model = model or self.default_model
        text = fixture if fixture is not None else prompt
        itok = max(1, len(prompt) // 4 + (len(system) // 4 if system else 0))
        otok = max(1, len(text) // 4)
        return Completion(
            text=text,
            usage=Usage(itok, otok, cost_of(model, itok, otok)),
            latency_ms=0.5,
            model=model,
            raw={"provider": "echo"},
        )


_REGISTRY: dict[str, Any] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "echo": EchoProvider,
}

# Friendlier spellings. Each maps to openai_compatible; a base_url is still
# required, these only save the user remembering the canonical name.
_ALIASES: dict[str, str] = {
    "openai-compatible": "openai_compatible",
    "openai_compat": "openai_compatible",
    "compatible": "openai_compatible",
    "custom": "openai_compatible",
    "local": "openai_compatible",
    "vllm": "openai_compatible",
    "ollama": "openai_compatible",
    "lmstudio": "openai_compatible",
    "lm_studio": "openai_compatible",
    "llamacpp": "openai_compatible",
    "openrouter": "openai_compatible",
    "together": "openai_compatible",
    "groq": "openai_compatible",
    "fireworks": "openai_compatible",
    "deepinfra": "openai_compatible",
}

# Convenience defaults for well-known self-hosted servers, applied only when the
# suite does not specify base_url / api_key_env itself.
_ALIAS_DEFAULTS: dict[str, dict[str, str]] = {
    "vllm": {"base_url": "http://localhost:8000/v1"},
    "ollama": {"base_url": "http://localhost:11434/v1"},
    "lmstudio": {"base_url": "http://localhost:1234/v1"},
    "lm_studio": {"base_url": "http://localhost:1234/v1"},
    "llamacpp": {"base_url": "http://localhost:8080/v1"},
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_API_KEY",
    },
}


def resolve_provider_name(name: str) -> str:
    key = (name or "").strip().lower()
    return _ALIASES.get(key, key)


def known_providers() -> list[str]:
    return sorted(_REGISTRY)


def known_aliases() -> dict[str, str]:
    return dict(_ALIASES)


def get_provider(name: str, **opts: Any):
    """Instantiate a provider.

    opts: base_url, api_key_env, api_key, extra_headers, timeout, max_retries.
    Alias defaults (e.g. ollama -> localhost:11434) fill only what the caller
    left unset.
    """
    raw = (name or "").strip().lower()
    canonical = resolve_provider_name(raw)
    cls = _REGISTRY.get(canonical)
    if cls is None:
        raise ProviderError(
            f"unknown provider {name!r}; built-in: {known_providers()}; "
            f"aliases: {sorted(_ALIASES)}"
        )
    clean = {k: v for k, v in opts.items() if v is not None}
    for k, v in _ALIAS_DEFAULTS.get(raw, {}).items():
        clean.setdefault(k, v)
    return cls(**clean)


def default_model_for(provider: str) -> str:
    cls = _REGISTRY.get(resolve_provider_name(provider))
    if cls is None:
        raise ProviderError(f"unknown provider {provider!r}")
    return getattr(cls, "default_model", "") or ""


def infer_provider(model: str) -> str:
    """Resolve a model id to a provider.

    Prefers the pricing table (data, updatable without touching code) and falls
    back to name prefixes. Never infers openai_compatible — that always needs an
    explicit base_url, so guessing it would produce a confusing connection error.
    """
    for provider in ("anthropic", "openai", "echo"):
        if lookup(model, provider) is not None:
            return provider

    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt", "davinci", "babbage")):
        return "openai"
    if m.startswith("echo"):
        return "echo"
    raise ProviderError(
        f"cannot infer provider for model {model!r}. Set `provider:` in the suite "
        f"(use `openai_compatible` with a `base_url:` for self-hosted or gateway "
        f"models), or add the model to the pricing table. "
        f"Known models include: {known_models()[:6]}"
    )


def dump_pricing() -> str:
    return render()
