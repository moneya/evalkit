"""Pricing lookup.

Prices live in data/pricing.json, not in code, because they change often and a
hardcoded table rots silently. Override order:

  1. $EVALKIT_PRICING            -> path to a replacement JSON file
  2. ./evalkit-pricing.json      -> project-local overrides, merged over defaults
  3. bundled data/pricing.json

An unknown model is NOT silently priced. `lookup` returns None, cost is recorded
as None, and the run is marked `unpriced` — because a fabricated cost number in a
tool whose whole purpose is cost gating is worse than no number at all.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_BUNDLED = Path(__file__).parent / "data" / "pricing.json"
_LOCAL_OVERRIDE = "evalkit-pricing.json"
_META_KEYS = {"_comment", "_verified", "_sources", "_schema"}


class PricingError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    try:
        doc = json.loads(_BUNDLED.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise PricingError(f"cannot read bundled pricing table {_BUNDLED}: {e}") from e

    env_path = os.environ.get("EVALKIT_PRICING", "").strip()
    if env_path:
        p = Path(env_path)
        if not p.exists():
            raise PricingError(f"EVALKIT_PRICING points at a missing file: {p}")
        doc = _merge(doc, json.loads(p.read_text()))

    local = Path.cwd() / _LOCAL_OVERRIDE
    if local.exists():
        try:
            doc = _merge(doc, json.loads(local.read_text()))
        except json.JSONDecodeError as e:
            raise PricingError(f"{local} is not valid JSON: {e}") from e

    return doc


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for provider, models in over.items():
        if provider in _META_KEYS:
            out[provider] = models
            continue
        if isinstance(models, dict) and isinstance(out.get(provider), dict):
            merged = dict(out[provider])
            merged.update(models)
            out[provider] = merged
        else:
            out[provider] = models
    return out


def reload() -> None:
    """Drop the cache — used by tests and after editing an override file."""
    _load.cache_clear()


def verified_on() -> str:
    return str(_load().get("_verified", "unknown"))


def sources() -> dict[str, str]:
    return dict(_load().get("_sources", {}))


def table() -> dict[str, dict[str, dict[str, float]]]:
    """Provider -> model -> {input, output}, metadata stripped."""
    return {k: v for k, v in _load().items() if k not in _META_KEYS}


def known_models(provider: str | None = None) -> list[str]:
    t = table()
    if provider:
        return sorted(t.get(provider, {}))
    return sorted(m for models in t.values() for m in models)


def lookup(model: str, provider: str | None = None) -> tuple[float, float] | None:
    """Exact-match price for a model, or None if we have no data for it.

    Deliberately exact: prefix guessing is how a pricing table quietly invents
    numbers. `gpt-5.5-pro` is 6x `gpt-5.5`; a prefix match would have reported
    the cheaper one and passed a cost gate it should have failed.
    """
    t = table()
    scopes = [provider] if provider else list(t)
    for p in scopes:
        entry = (t.get(p) or {}).get(model)
        if entry:
            return float(entry["input"]), float(entry["output"])
    return None


def cost_of(model: str, input_tokens: int, output_tokens: int,
            provider: str | None = None) -> float | None:
    """USD for one call, or None when the model is unpriced."""
    price = lookup(model, provider)
    if price is None:
        return None
    pin, pout = price
    return round((input_tokens * pin + output_tokens * pout) / 1_000_000, 8)


def render() -> str:
    lines = [f"pricing verified {verified_on()}  (USD per 1M tokens)"]
    for src, url in sources().items():
        lines.append(f"  source[{src}]: {url}")
    for provider, models in sorted(table().items()):
        lines.append(f"\n{provider}")
        for m, p in sorted(models.items(), key=lambda kv: -kv[1]["output"]):
            lines.append(f"  {m:<30} in ${p['input']:>7.3f}   out ${p['output']:>8.3f}")
    lines.append(
        "\nOverride with $EVALKIT_PRICING=/path/to.json or ./evalkit-pricing.json"
    )
    return "\n".join(lines)
