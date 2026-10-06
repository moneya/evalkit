"""Model providers. Each returns a Completion with real token counts and cost.

Built in: anthropic, openai, echo (offline, deterministic — used by tests and
by `evalkit run --dry-run` so the harness is usable with no API key).

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
    "EchoProvider",
    "ProviderError",
    "get_provider",
    "infer_provider",
    "default_model_for",
    "dump_pricing",
]


class ProviderError(RuntimeError):
    pass


class Provider(Protocol):
    name: str

    def complete(
        self, prompt: str, *, system: str | None, model: str, max_tokens: int, temperature: float
    ) -> Completion: ...


class _HTTPProvider:
    """Shared retry/transport behaviour for HTTP providers."""

    name = "http"
    env_key = ""
    timeout = 120.0
    max_retries = 3

    def _api_key(self) -> str:
        key = os.environ.get(self.env_key, "").strip()
        if not key:
            raise ProviderError(
                f"{self.env_key} is not set — export it, or run with --provider echo."
            )
        return key

    def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    r = client.post(url, headers=headers, json=payload)
                if r.status_code in (429, 500, 502, 503, 529):
                    raise httpx.HTTPStatusError(
                        f"{r.status_code} {r.text[:200]}", request=r.request, response=r
                    )
                if r.status_code >= 400:
                    raise ProviderError(f"{self.name} {r.status_code}: {r.text[:400]}")
                return r.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(1.5 * (2**attempt))
        raise ProviderError(f"{self.name} failed after {self.max_retries} attempts: {last}")

    def _usage(self, model: str, itok: int, otok: int) -> Usage:
        return Usage(
            input_tokens=itok,
            output_tokens=otok,
            cost_usd=cost_of(model, itok, otok, provider=self.name),
        )


class AnthropicProvider(_HTTPProvider):
    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"
    default_model = "claude-haiku-4-5"
    url = "https://api.anthropic.com/v1/messages"
    api_version = "2023-06-01"

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
        headers = {
            "x-api-key": self._api_key(),
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }
        t0 = time.perf_counter()
        data = self._post(self.url, headers, payload)
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
    default_model = "gpt-5-mini"
    url = "https://api.openai.com/v1/chat/completions"

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
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        # Reasoning models reject an explicit temperature, so only send one when
        # the suite asked for something other than the provider default.
        if not _is_reasoning_model(model):
            payload["temperature"] = temperature

        headers = {
            "authorization": f"Bearer {self._api_key()}",
            "content-type": "application/json",
        }
        t0 = time.perf_counter()
        data = self._post(self.url, headers, payload)
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


class EchoProvider:
    """Offline provider. Deterministic, free, no network.

    Reply resolution:
      1. `fixture` on the case (exact canned output)
      2. the prompt echoed back

    Token counts are a ~4-chars-per-token approximation, not a real tokenizer.
    Cost follows the pricing table for the given model id, so naming a priced
    model (e.g. gpt-4o-mini) gives realistic cost arithmetic with no network
    call — that is how the regression-gate demo stays free and deterministic.
    """

    name = "echo"
    default_model = "echo-1"

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
    "echo": EchoProvider,
}


def get_provider(name: str):
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ProviderError(f"unknown provider {name!r}; have {sorted(_REGISTRY)}")
    return cls()


def default_model_for(provider: str) -> str:
    return get_provider(provider).default_model


def infer_provider(model: str) -> str:
    """Resolve a model id to a provider.

    Prefers the pricing table (data, updatable without touching code) and falls
    back to name prefixes for models not yet listed there.
    """
    for provider in _REGISTRY:
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
        f"cannot infer provider for model {model!r}. Set `provider:` in the suite, "
        f"or add the model to the pricing table. Known models include: {known_models()[:6]}"
    )


def dump_pricing() -> str:
    return render()
