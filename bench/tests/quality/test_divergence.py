import math

import pytest

from loom_bench.quality.divergence import (
    Position,
    aggregate,
    approx_kl,
    compare_positions,
    load_prompts,
    scored_positions,
    top1,
)

ln = math.log


def test_kl_identical_is_zero():
    d = {"a": ln(0.6), "b": ln(0.3)}
    assert approx_kl(d, d) == pytest.approx(0.0, abs=1e-12)


def test_kl_same_support_matches_hand_computation():
    # P = (.7, .2, other .1), Q = (.6, .3, other .1)
    ref = {"a": ln(0.7), "b": ln(0.2)}
    cand = {"a": ln(0.6), "b": ln(0.3)}
    expected = 0.7 * ln(0.7 / 0.6) + 0.2 * ln(0.2 / 0.3) + 0.1 * ln(0.1 / 0.1)
    assert approx_kl(ref, cand) == pytest.approx(expected, rel=1e-6)


def test_kl_imputes_missing_tokens_from_unlisted_mass():
    # Each side misses one token of the union; it gets min(unlisted / 2, smallest listed)
    # = 0.1 and the other 0.1 stays in "other": P = (.5, .3, .1, .1), Q = (.5, .1, .3, .1).
    ref = {"a": ln(0.5), "b": ln(0.3)}
    cand = {"a": ln(0.5), "c": ln(0.3)}
    expected = 0.3 * ln(3) + 0.1 * ln(1 / 3)
    assert approx_kl(ref, cand) == pytest.approx(expected, rel=1e-6)


def test_kl_imputed_mass_is_capped_by_smallest_listed():
    # Unlisted mass 0.5, one missing token: share = min(0.25, 0.1) = 0.1, other = 0.4.
    ref = {"a": ln(0.4), "b": ln(0.1)}
    cand = {"a": ln(0.4), "b": ln(0.1), "c": ln(0.1)}
    p = [0.4, 0.1, 0.1, 0.4]
    q = [0.4, 0.1, 0.1, 0.4]
    expected = sum(x * ln(x / y) for x, y in zip(p, q, strict=True))
    assert approx_kl(ref, cand) == pytest.approx(expected, abs=1e-9)


def test_kl_disjoint_confident_tokens_is_large():
    ref = {"a": ln(0.99)}
    cand = {"b": ln(0.99)}
    # P = (.99, .005, other .005), Q = (.005, .99, other .005)
    expected = 0.99 * ln(0.99 / 0.005) + 0.005 * ln(0.005 / 0.99)
    assert approx_kl(ref, cand) == pytest.approx(expected, rel=1e-6)
    assert approx_kl(ref, cand) > 5


def test_kl_needs_listed_tokens():
    with pytest.raises(ValueError):
        approx_kl({}, {"a": 0.0})


def test_top1_breaks_ties_deterministically():
    assert top1({"b": -0.1, "a": -0.1, "c": -2.0}) == "a"
    assert top1({"x": -3.0, "y": -0.5}) == "y"


ECHO = {
    # prompt "Hi" + continuation " there you", then one generated token at offset 10.
    "tokens": ["Hi", " there", " you", " go"],
    "token_logprobs": [None, -0.5, -0.2, -0.1],
    "top_logprobs": [None, {" there": -0.5, " all": -1.2}, {" you": -0.2}, {" go": -0.1}],
    "text_offset": [0, 2, 8, 12],
}


def test_scored_positions_keep_only_the_continuation():
    pos = scored_positions(ECHO, 2, 12)
    assert [p.token for p in pos] == [" there", " you"]
    assert pos[0].top == {" there": -0.5, " all": -1.2}


def test_scored_positions_rebuild_offsets_a_server_reports_as_unknown():
    # SGLang returns text_offset -1 for every token; cumulative token lengths give
    # vLLM's offsets back, so the same continuation positions are selected.
    sglang = {**ECHO, "text_offset": [-1, -1, -1, -1]}
    pos = scored_positions(sglang, 2, 12, text="Hi there you")
    assert [p.token for p in pos] == [" there", " you"]
    assert pos == scored_positions(ECHO, 2, 12, text="Hi there you")


def test_scored_positions_reject_partly_unknown_offsets():
    with pytest.raises(ValueError, match="mixes"):
        scored_positions({**ECHO, "text_offset": [0, -1, 8, 12]}, 2, 12)


def test_scored_positions_reject_rebuilt_offsets_that_do_not_spell_the_text():
    sglang = {**ECHO, "text_offset": [-1, -1, -1, -1]}
    with pytest.raises(ValueError, match="do not spell"):
        scored_positions(sglang, 2, 12, text="Hi their you")


def test_scored_positions_reject_missing_top_logprobs():
    bad = {**ECHO, "top_logprobs": [None, None, {" you": -0.2}, {" go": -0.1}]}
    with pytest.raises(ValueError):
        scored_positions(bad, 2, 12)


def test_compare_positions():
    ref = [Position("x", {"x": ln(0.9), "y": ln(0.05)}), Position("y", {"y": ln(0.6)})]
    cand = [Position("x", {"y": ln(0.7), "x": ln(0.2)}), Position("y", {"y": ln(0.6)})]
    d = compare_positions(ref, cand)
    assert d.agree == [False, True]
    assert d.top1_rate == 0.5
    assert d.kl[0] > 0.5
    assert d.kl[1] == pytest.approx(0.0, abs=1e-12)


def test_compare_positions_requires_same_tokenization():
    with pytest.raises(ValueError, match="tokenized"):
        compare_positions([Position("a", {"a": 0.0})], [Position("b", {"b": 0.0})])


def test_aggregate_over_prompts():
    from loom_bench.quality.divergence import PromptDivergence

    prompts = [PromptDivergence(kl=[0.1, 0.3], agree=[True, False]) for _ in range(5)]
    prompts.append(PromptDivergence(kl=[0.0], agree=[True]))
    res = aggregate(prompts, top_k=5, n_skipped=1, n_boot=500)
    assert res.n_prompts == 6 and res.n_positions == 11 and res.n_skipped == 1
    assert res.kl.point == pytest.approx((5 * 0.2 + 0.0) / 6)
    assert res.top1.point == pytest.approx((5 * 0.5 + 1.0) / 6)
    assert res.kl.lo <= res.kl.point <= res.kl.hi


def test_pinned_prompts_load():
    prompts = load_prompts()
    assert len(prompts) >= 40
    assert len(set(prompts)) == len(prompts)
