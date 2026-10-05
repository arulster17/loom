from fractions import Fraction

import pytest

from loom_bench.money import (
    cost_for_seconds,
    cost_for_tokens,
    cost_per_mtok,
    format_usd,
    parse_usd,
    round_half_up,
)


def test_parse_usd_variants():
    assert parse_usd("$1.861") == 1_861_000
    assert parse_usd("0.000001") == 1
    assert parse_usd(" 50 ") == 50_000_000
    assert parse_usd(1_234) == 1_234  # ints are already micros


def test_parse_usd_rejects_sub_micro_and_garbage():
    with pytest.raises(ValueError):
        parse_usd("0.0000001")
    with pytest.raises(ValueError):
        parse_usd("abc")
    with pytest.raises(TypeError):
        parse_usd(True)  # type: ignore[arg-type]


def test_round_half_up_is_symmetric():
    assert round_half_up(Fraction(5, 2)) == 3
    assert round_half_up(Fraction(-5, 2)) == -3
    assert round_half_up(Fraction(7, 3)) == 2


def test_cost_per_mtok():
    # $3.60/h at 1000 tok/s -> 3.6M tok/h -> $1.00 per 1M tokens
    assert cost_per_mtok(3_600_000, 1000) == 1_000_000
    assert cost_per_mtok(3_600_000, 0) is None


def test_cost_for_seconds_and_tokens():
    assert cost_for_seconds(3_600_000, 1) == 1_000  # $3.60/h for 1s = $0.001
    assert cost_for_tokens(150_000, 2_000_000) == 300_000
    assert cost_for_tokens(1, 1) == 0  # rounds once, at the end


def test_format_usd():
    assert format_usd(1_861_000) == "$1.8610"
    assert format_usd(-500_000, places=2) == "-$0.50"
