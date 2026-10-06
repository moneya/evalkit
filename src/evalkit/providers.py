"""Model providers. Each returns a Completion with real token counts and cost.

Built in: anthropic, openai, echo (offline, deterministic — used by tests and
by `evalkit run --dry-run` so the harness is usable with no API key).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol

import httpx

from .types import Completion, Usage

# USD per 1M tokens (input, output). Update as pricing changes.
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "o4-mini": (1.10, 4.40),
}

DEFAULT_PRICE = (1.00, 3.00)


def price_for(model: str) -> tuple[float, float]:
    """Exact match, else longest known prefix, else a flat default."""
    if model in PRICING:
        return PRICING[model]
    best, best_len = DEFAULT_PRICE, 0
    for known, p in PRICING.items():
        stem = known.rsplit("-", 1)[0]
        if model.startswith(stem) and len(stem) > best_len:
            best, best_len = p, len(stem)
    return best


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = price_for(model)
    return round((input_tokens * pin + output_tokens * pout) / 1_000_000, 8)


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


class AnthropicProvider(_HTTPProvider):
    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"
    default_model = "claude-3-5-haiku-20241022"
    url = "https://api.anthropic.com/v1/messages"

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
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        t0 = time.perf_counter()
        data = self._post(self.url, headers, payload)
        latency = (time.perf_counter() - t0) * 1000

        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage", {})
        itok, otok = int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
        return Completion(
            text=text,
            usage=Usage(itok, otok, cost_of(model, itok, otok)),
            latency_ms=latency,
            model=model,
            raw=data,
        )


class OpenAIProvider(_HTTPProvider):
    name = "openai"
    env_key = "OPENAI_API_KEY"
    default_model = "gpt-4o-mini"
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
        payload = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
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
            usage=Usage(itok, otok, cost_of(model, itok, otok)),
            latency_ms=latency,
            model=model,
            raw=data,
        )


class EchoProvider:
    """Offline provider. Deterministic, free, no network.

    Resolution order for the reply:
      1. `fixture` on the case (exact canned output)
      2. the case's `expect` field, if present
      3. the prompt echoed back

    This makes the harness and its own test suite runnable with zero API keys.
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
            usage=Usage(itok, otok, cost_of(model, itok, otok) if model in PRICING else 0.0),
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


def infer_provider(model: str) -> str:
    """Guess the provider from a model id."""
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("echo"):
        return "echo"
    raise ProviderError(f"cannot infer provider for model {model!r}; set `provider:` explicitly")


def dump_pricing() -> str:
    return json.dumps(PRICING, indent=2, sort_keys=True)
