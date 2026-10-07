"""Tests for the pricing table — the component most likely to rot.

The failure mode these guard against: a hardcoded table that silently invents a
price for an unknown model, so a cost gate passes on a fabricated number.
"""

from __future__ import annotations

import json

import pytest

from evalkit import pricing
from evalkit.assertions import run_assertions
from evalkit.types import Completion, Usage


@pytest.fixture(autouse=True)
def _fresh_table():
    pricing.reload()
    yield
    pricing.reload()


# -- no silent guessing ---------------------------------------------------

def test_unknown_model_returns_none_not_a_default():
    assert pricing.lookup("totally-made-up-model") is None
    assert pricing.cost_of("totally-made-up-model", 1000, 1000) is None


def test_lookup_is_exact_not_prefix_based():
    """Prefix matching is how a table under-reports a pricier sibling."""
    assert pricing.lookup("gpt-5.5") == (5.00, 30.00)
    assert pricing.lookup("gpt-5.5-pro") == (30.00, 180.00)
    assert pricing.lookup("gpt-5.5-turbo-imaginary") is None


def test_provider_scoped_lookup_rejects_cross_provider_hit():
    assert pricing.lookup("gpt-4o-mini", provider="openai") is not None
    assert pricing.lookup("gpt-4o-mini", provider="anthropic") is None


# -- cost arithmetic ------------------------------------------------------

def test_cost_math_matches_published_rate():
    # claude-sonnet-5-5: $2 in / $10 out per 1M tokens
    assert pricing.cost_of("claude-sonnet-5-5", 1_000_000, 0) == pytest.approx(2.00)
    assert pricing.cost_of("claude-sonnet-5-5", 0, 1_000_000) == pytest.approx(10.00)
    assert pricing.cost_of("claude-sonnet-5-5", 500_000, 100_000) == pytest.approx(2.0)


def test_cost_is_zero_for_zero_tokens():
    assert pricing.cost_of("gpt-5-mini", 0, 0) == 0.0


def test_echo_model_is_free_but_priced():
    assert pricing.lookup("echo-1") == (0.0, 0.0)
    assert pricing.cost_of("echo-1", 10_000, 10_000) == 0.0


# -- table integrity ------------------------------------------------------

def test_table_has_metadata_and_sources():
    assert pricing.verified_on() != "unknown"
    src = pricing.sources()
    assert "anthropic" in src and src["anthropic"].startswith("http")
    assert "openai" in src


def test_metadata_keys_excluded_from_model_table():
    for provider, models in pricing.table().items():
        assert not provider.startswith("_")
        for m in models:
            assert not m.startswith("_")


def test_every_entry_is_well_formed():
    for provider, models in pricing.table().items():
        for model, p in models.items():
            keys = set(p)
            assert {"input", "output"} <= keys, f"{provider}/{model}"
            # `note` is the only optional extra: it documents what two numbers
            # cannot, e.g. that a vendor's rate varies by time of day.
            assert keys <= {"input", "output", "note"}, f"{provider}/{model}: {keys}"
            if "note" in p:
                assert isinstance(p["note"], str) and p["note"]
            assert p["input"] >= 0 and p["output"] >= 0, f"{provider}/{model}"
            if p["output"] > 0:
                assert p["output"] >= p["input"], f"{provider}/{model}: output cheaper than input"


def test_known_models_is_non_trivial_and_scopable():
    assert len(pricing.known_models()) >= 30
    assert "claude-haiku-4-5" in pricing.known_models("anthropic")
    assert "claude-haiku-4-5" not in pricing.known_models("openai")


def test_provider_defaults_are_actually_priced():
    """A default model that isn't in the table would fail every cost gate."""
    from evalkit.providers import default_model_for

    for provider in ("anthropic", "openai", "echo"):
        model = default_model_for(provider)
        assert pricing.lookup(model, provider) is not None, f"{provider} default {model} unpriced"


# -- overrides ------------------------------------------------------------

def test_env_override_merges_over_defaults(tmp_path, monkeypatch):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"openai": {"gpt-4o-mini": {"input": 99.0, "output": 100.0}}}))
    monkeypatch.setenv("EVALKIT_PRICING", str(f))
    pricing.reload()
    assert pricing.lookup("gpt-4o-mini") == (99.0, 100.0)
    # untouched entries survive the merge
    assert pricing.lookup("claude-haiku-4-5") == (1.00, 5.00)


def test_env_override_can_add_a_new_model(tmp_path, monkeypatch):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"openai": {"gpt-7-future": {"input": 1.0, "output": 2.0}}}))
    monkeypatch.setenv("EVALKIT_PRICING", str(f))
    pricing.reload()
    assert pricing.cost_of("gpt-7-future", 1_000_000, 0) == pytest.approx(1.0)


def test_missing_env_override_is_a_loud_error(monkeypatch):
    monkeypatch.setenv("EVALKIT_PRICING", "/nope/does/not/exist.json")
    pricing.reload()
    with pytest.raises(pricing.PricingError, match="missing file"):
        pricing.lookup("gpt-4o")


def test_render_includes_verification_date_and_sources():
    out = pricing.render()
    assert "verified" in out
    assert "claude-haiku-4-5" in out
    assert "EVALKIT_PRICING" in out


# -- the gate must not pass on an unverifiable cost -----------------------

def test_max_cost_fails_closed_on_unpriced_model():
    c = Completion(
        text="hi",
        usage=Usage(10, 10, cost_usd=None),
        latency_ms=10.0,
        model="mystery-model-1",
    )
    r = run_assertions([{"max_cost": 0.01}], c)
    assert not r[0].passed
    assert "not in the pricing table" in r[0].detail


def test_time_varying_prices_store_the_higher_rate():
    """A cost gate may over-estimate safely; under-estimating passes regressions.

    DeepSeek charges about 2x at peak vs off-peak. We store peak, and the note
    must document the cheaper tier, otherwise a reader cannot tell which tier
    the stored number represents.
    """
    deepseek = pricing.table()["deepseek"]
    for model in ("deepseek-flash", "deepseek-v4-pro"):
        entry = deepseek[model]
        assert "off-peak" in entry["note"], f"{model} must document the cheaper tier"
        off_peak_input = float(entry["note"].split("$")[1].split("/")[0])
        assert entry["input"] > off_peak_input, f"{model} stored off-peak, not peak"


def test_deepseek_models_are_priced_and_resolvable():
    for model in ("deepseek-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"):
        assert pricing.lookup(model) is not None, model
        assert model in pricing.known_models("deepseek")
