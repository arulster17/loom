import itertools
from fractions import Fraction
from math import ceil

import pytest

from loom_bench.registry import load_registry
from loom_platform.pricing import (
    TEMPLATE_OVERHEAD_TOKENS,
    Price,
    UnpricedModel,
    cost_micros,
    max_cost_micros,
    price_for,
    prompt_tokens_upper_bound,
    servable,
)

QWEN = Price(77_000, 77_000, 368_000)  # config/models.yaml qwen3-8b


def test_cost_examples():
    # 1,000 prompt + 500 completion on Qwen3 8B: 77 + 184 = 261 micros exactly.
    assert cost_micros(QWEN, 1000, 500) == 261
    # One prompt token is 0.077 micros: rounds up to 1, never down to 0.
    assert cost_micros(QWEN, 1, 0) == 1
    assert cost_micros(QWEN, 0, 0) == 0
    # 1M of each side is exactly the list prices.
    assert cost_micros(QWEN, 1_000_000, 1_000_000) == 77_000 + 368_000


def test_cost_rounds_up_once_on_the_sum():
    # 3 prompt tokens (0.231) + 1 completion (0.368) = 0.599 -> 1, not 1 + 1.
    assert cost_micros(QWEN, 3, 1) == 1
    p = Price(1_500_000, 500_000, 2_500_000)
    for prompt, completion, cached in itertools.product(range(9), range(9), range(9)):
        if cached > prompt:
            continue
        exact = Fraction(
            (prompt - cached) * 1_500_000 + cached * 500_000 + completion * 2_500_000, 1_000_000
        )
        got = cost_micros(p, prompt, completion, cached)
        assert got == ceil(exact)
        assert 0 <= got - exact < 1


def test_cached_tokens_priced_at_cached_rate():
    p = Price(1_000_000, 250_000, 2_000_000)
    assert cost_micros(p, 10, 0, cached_prompt_tokens=0) == 10
    assert cost_micros(p, 10, 0, cached_prompt_tokens=8) == 2 + 2  # 2 x 1 + 8 x 0.25
    with pytest.raises(ValueError):
        cost_micros(p, 5, 0, cached_prompt_tokens=6)


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_cost_rejects_bad_counts(bad):
    with pytest.raises(ValueError):
        cost_micros(QWEN, bad, 0)


def test_price_validation_and_version_round_trip():
    assert QWEN.version == "77000:77000:368000"
    assert Price.from_version(QWEN.version) == QWEN
    with pytest.raises(ValueError):
        Price(1, 2, 3)  # cached above input
    with pytest.raises(ValueError):
        Price(-1, 0, 0)
    with pytest.raises(ValueError):
        Price.from_version("1.5:1:1")


def test_registry_prices():
    reg = load_registry()
    assert price_for(reg, "qwen3-8b") == QWEN
    assert price_for(reg, "llama-3.3-70b-instruct-fp8") == Price(571_000, 571_000, 797_000)
    with pytest.raises(UnpricedModel):
        price_for(reg, "llama-3.3-70b-instruct")  # BF16 reference row, unpriced
    with pytest.raises(KeyError):
        price_for(reg, "nope")
    assert {m.id for m in servable(reg)} == {"qwen3-8b", "llama-3.3-70b-instruct-fp8"}


def test_prompt_upper_bound():
    assert prompt_tokens_upper_bound(100, 32768) == 100 + TEMPLATE_OVERHEAD_TOKENS
    assert prompt_tokens_upper_bound(10**6, 32768) == 32768
    assert prompt_tokens_upper_bound(100, 32768, template_json=True) == 32768


def test_max_cost_covers_every_feasible_outcome():
    """Brute force: nothing the engine can return costs more than the reservation, for
    output-dearer, input-dearer and equal prices; and the bound is tight."""
    for price in (Price(3, 1, 7), Price(9, 2, 4), Price(5, 5, 5), QWEN):
        for p_up, c_max, ctx in itertools.product((0, 3, 10), (0, 4, 12), (1, 8, 15)):
            reserved = max_cost_micros(price, p_up, c_max, ctx)
            worst = 0
            for p in range(min(p_up, ctx) + 1):
                for c in range(min(c_max, ctx - p) + 1):
                    worst = max(worst, cost_micros(price, p, c))
                    assert cost_micros(price, p, c, cached_prompt_tokens=p) <= reserved
            assert reserved == worst


def test_max_cost_realistic():
    # Qwen3 8B, 2 KB body, max_tokens 1024, 32k context.
    p_up = prompt_tokens_upper_bound(2048, 32768)
    assert max_cost_micros(QWEN, p_up, 1024, 32768) == ceil(
        Fraction(p_up * 77_000 + 1024 * 368_000, 1_000_000)
    )
