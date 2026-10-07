import json

import numpy as np
import pytest

from loom_bench.quality.divergence import DivergenceResult
from loom_bench.quality.gate import (
    GatePolicy,
    TaskPolicy,
    Verdict,
    divergence_limits,
    evaluate_gate,
    evaluate_task,
    worst,
)
from loom_bench.quality.sanity import SanityLimits, SanityResult
from loom_bench.quality.tasks.base import ItemResult
from loom_bench.stats import Interval

FAST = GatePolicy(n_boot=2000, min_samples=50)


def items(scores, prefix="q", hashes=None):
    return [
        ItemResult(
            item_id=f"{prefix}{i}", score=s, content_hash=None if hashes is None else hashes[i]
        )
        for i, s in enumerate(scores)
    ]


def binary(n, p, seed):
    return (np.random.default_rng(seed).random(n) < p).astype(float).tolist()


def flip(scores, k_down, k_up=0):
    """Copy of `scores` with the first k_down ones turned to 0 and first k_up zeros to 1."""
    out = list(scores)
    ones = [i for i, s in enumerate(out) if s == 1.0][:k_down]
    zeros = [i for i, s in enumerate(out) if s == 0.0][:k_up]
    for i in ones:
        out[i] = 0.0
    for i in zeros:
        out[i] = 1.0
    return out


def test_identical_large_sample_passes():
    base = binary(1000, 0.8, 1)
    v = evaluate_task("t", items(base), items(base), FAST)
    assert v.verdict is Verdict.PASS
    assert v.delta == 0.0
    # Rule-of-three floor: a collapsed bootstrap CI is widened to ±3/n.
    assert v.bootstrap_low == v.bootstrap_high == 0.0
    assert v.ci_low == pytest.approx(-0.003)
    assert v.ci_high == pytest.approx(0.003)


def test_symmetric_noise_passes():
    base = binary(2000, 0.7, 2)
    cand = flip(base, 10, 10)  # 1% of items churn both ways, net zero
    v = evaluate_task("t", items(base), items(cand), FAST)
    assert v.verdict is Verdict.PASS
    assert (v.improved, v.regressed) == (10, 10)
    assert v.ci_low >= -0.01


def test_clear_drop_fails():
    base = binary(1000, 0.8, 3)
    cand = flip(base, 50)
    v = evaluate_task("t", items(base), items(cand), FAST)
    assert v.verdict is Verdict.FAIL
    assert v.delta == pytest.approx(-0.05)
    assert v.ci_high < -0.01
    assert "drop" in v.reason


def test_point_drop_beyond_threshold_fails_even_with_wide_ci():
    base = binary(400, 0.5, 4)
    cand = flip(base, 30, 24)  # net -1.5 pts with lots of churn
    v = evaluate_task("t", items(base), items(cand), FAST)
    assert v.delta == pytest.approx(-0.015)
    assert v.ci_high > -0.01
    assert v.verdict is Verdict.FAIL


def test_small_drop_with_wide_ci_is_inconclusive():
    base = binary(400, 0.5, 5)
    cand = flip(base, 22, 20)  # net -0.5 pts, CI spans the -1 pt margin
    v = evaluate_task("t", items(base), items(cand), FAST)
    assert v.delta == pytest.approx(-0.005)
    assert v.ci_low < -0.01 < v.ci_high
    assert v.verdict is Verdict.INCONCLUSIVE
    assert "more samples" in v.reason


def test_identical_small_sample_is_inconclusive_by_resolution_floor():
    base = binary(150, 0.8, 6)
    v = evaluate_task("t", items(base), items(base), FAST)  # min_samples=50 is met
    assert v.bootstrap_low == 0.0
    assert v.ci_low == pytest.approx(-0.02)
    assert v.verdict is Verdict.INCONCLUSIVE


def test_below_min_samples_is_inconclusive_even_for_big_drops():
    base = [1.0] * 40
    cand = [0.0] * 40
    v = evaluate_task("t", items(base), items(cand), FAST)
    assert v.verdict is Verdict.INCONCLUSIVE
    assert "min_samples" in v.reason
    assert v.ci_low is None


def test_per_task_threshold_and_min_samples():
    base = binary(400, 0.5, 7)
    cand = flip(base, 12)  # -3 pts
    policy = FAST.model_copy(update={"tasks": {"t": TaskPolicy(threshold=0.1, min_samples=10)}})
    v = evaluate_task("t", items(base), items(cand), policy)
    assert v.threshold == 0.1
    assert v.verdict is Verdict.PASS


def test_improvement_passes():
    base = binary(1000, 0.6, 8)
    v = evaluate_task("t", items(base), items(flip(base, 0, 40)), FAST)
    assert v.delta == pytest.approx(0.04)
    assert v.verdict is Verdict.PASS


def test_pairing_is_by_item_id_not_order():
    base = items(binary(500, 0.7, 9))
    v = evaluate_task("t", base, list(reversed(base)), FAST)
    assert v.delta == 0.0
    assert v.improved == v.regressed == 0


def test_mismatched_items_raise():
    base = items([1.0] * 100)
    with pytest.raises(ValueError, match="different items"):
        evaluate_task("t", base, items([1.0] * 100, prefix="x"), FAST)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_task("t", base + base[:1], base + base[:1], FAST)
    h1 = items([1.0] * 100, hashes=[str(i) for i in range(100)])
    h2 = items([1.0] * 100, hashes=[str(i + 1) for i in range(100)])
    with pytest.raises(ValueError, match="different content"):
        evaluate_task("t", h1, h2, FAST)


def _div(kl, top1):
    def iv(x):
        return Interval(point=x, lo=x, hi=x, n=40, confidence=0.95, n_boot=100, seed=0)

    return DivergenceResult(
        n_prompts=40, n_positions=2000, n_skipped=0, top_k=5, kl=iv(kl), top1=iv(top1)
    )


def test_worst_task_decides_and_blocks():
    good = binary(1000, 0.8, 10)
    bad = binary(1000, 0.8, 11)
    d = evaluate_gate(
        {"a": items(good), "b": items(bad)},
        {"a": items(good), "b": items(flip(bad, 60))},
        None,
        None,
        FAST,
    )
    assert [t.verdict for t in d.tasks] == [Verdict.PASS, Verdict.FAIL]
    assert d.decision is Verdict.FAIL and d.blocked
    assert d.divergence.reason == "not measured"


def test_inconclusive_blocks_unless_disabled():
    base = {"a": items([1.0] * 10)}
    d = evaluate_gate(base, base, None, None, FAST)
    assert d.decision is Verdict.INCONCLUSIVE and d.blocked
    lenient = FAST.model_copy(update={"inconclusive_blocks": False})
    d = evaluate_gate(base, base, None, None, lenient)
    assert d.decision is Verdict.INCONCLUSIVE and not d.blocked


def _floor(kl_hi, top1_lo, *, kl=None, top1=None):
    """A self-divergence whose CI bounds are what the gate calibrates on."""

    def iv(point, lo, hi):
        return Interval(point=point, lo=lo, hi=hi, n=40, confidence=0.95, n_boot=100, seed=0)

    kl = kl_hi / 2 if kl is None else kl
    top1 = (1 + top1_lo) / 2 if top1 is None else top1
    return DivergenceResult(
        n_prompts=40,
        n_positions=2000,
        n_skipped=0,
        top_k=5,
        kl=iv(kl, 0.0, kl_hi),
        top1=iv(top1, top1_lo, 1.0),
    )


PASSING = {"a": items(binary(1000, 0.8, 12))}


def gate(div, floor=None, policy=FAST, candidate=None):
    return evaluate_gate(PASSING, candidate or PASSING, div, None, policy, self_divergence=floor)


def test_calibrated_limits_widen_to_the_noise_multiple():
    lim = divergence_limits(FAST, _floor(0.02, 0.98))
    assert lim.calibrated
    assert (lim.self_kl, lim.self_top1) == (0.02, 0.98)
    assert lim.max_kl == pytest.approx(5 * 0.02)  # above the absolute 0.05
    assert lim.min_top1 == pytest.approx(1 - 5 * 0.02)  # disagreement 2% -> 10%
    assert (lim.ceiling_kl, lim.ceiling_top1) == (0.5, 0.8)


def test_calibrated_limits_never_stricter_than_the_absolute_ones():
    lim = divergence_limits(FAST, _floor(0.001, 0.999))
    assert lim.calibrated
    assert lim.max_kl == 0.05 and lim.min_top1 == 0.95
    zero = divergence_limits(FAST, _floor(0.0, 1.0))
    assert zero.max_kl == 0.05 and zero.min_top1 == 0.95


def test_calibration_uses_the_conservative_ci_bound():
    lim = divergence_limits(FAST, _floor(0.03, 0.97, kl=0.01, top1=0.995))
    assert lim.max_kl == pytest.approx(0.15) and lim.min_top1 == pytest.approx(0.85)
    tripled = divergence_limits(FAST.model_copy(update={"noise_multiple": 3.0}), _floor(0.03, 0.97))
    assert tripled.max_kl == pytest.approx(0.09) and tripled.min_top1 == pytest.approx(0.91)


def test_divergence_within_the_calibrated_limits_passes():
    # Above the old fixed 0.05 / 95%, inside 5x a 0.02 / 98% floor.
    d = gate(_div(0.08, 0.92), _floor(0.02, 0.98))
    assert d.divergence.verdict is Verdict.PASS
    assert d.decision is Verdict.PASS and not d.blocked
    assert d.divergence.limits.calibrated
    assert "noise floor" in d.divergence.reason


def test_divergence_beyond_the_limits_with_passing_tasks_needs_review():
    d = gate(_div(0.2, 0.92), _floor(0.02, 0.98))
    assert d.divergence.verdict is Verdict.REVIEW
    assert d.decision is Verdict.REVIEW and not d.blocked
    assert "needs review" in d.divergence.reason
    assert "0.2000 nats exceeds 0.1000" in d.divergence.reason
    assert "self-KL <= 0.0200" in d.divergence.reason
    assert "(needs review, not blocking)" in d.summary()
    low_top1 = gate(_div(0.01, 0.85), _floor(0.02, 0.98))
    assert low_top1.decision is Verdict.REVIEW
    assert "top-1 agreement 85.00% is below 90.00%" in low_top1.divergence.reason


def test_review_blocks_when_the_policy_says_so():
    strict = FAST.model_copy(update={"review_blocks": True})
    d = gate(_div(0.2, 0.92), _floor(0.02, 0.98), policy=strict)
    assert d.decision is Verdict.REVIEW and d.blocked
    assert "(blocked)" in d.summary()


def test_divergence_beyond_the_limits_with_a_failing_task_fails():
    base = PASSING["a"]
    dropped = {"a": items(flip([s.score for s in base], 60))}
    d = gate(_div(0.2, 0.92), _floor(0.02, 0.98), candidate=dropped)
    assert d.tasks[0].verdict is Verdict.FAIL
    assert d.divergence.verdict is Verdict.FAIL
    assert "a did not pass non-inferiority" in d.divergence.reason


def test_divergence_beyond_the_limits_with_an_inconclusive_task_fails():
    small = {"a": items([1.0] * 10)}
    d = evaluate_gate(small, small, _div(0.2, 0.92), None, FAST, self_divergence=_floor(0.02, 0.98))
    assert d.tasks[0].verdict is Verdict.INCONCLUSIVE
    assert d.divergence.verdict is Verdict.FAIL
    assert d.decision is Verdict.FAIL and d.blocked


def test_hard_ceiling_fails_whatever_the_floor():
    noisy = _floor(0.2, 0.7)  # a floor so high its 5x limits are beyond the ceiling
    for div, what in ((_div(0.6, 0.99), "KL"), (_div(0.01, 0.75), "top-1")):
        d = gate(div, noisy)
        assert d.divergence.verdict is Verdict.FAIL, what
        assert "hard ceiling" in d.divergence.reason and what in d.divergence.reason
        assert d.decision is Verdict.FAIL and d.blocked


def test_uncalibrated_divergence_uses_the_absolute_limits_and_says_so():
    ok = gate(_div(0.01, 0.99))
    assert ok.decision is Verdict.PASS
    assert "uncalibrated" in ok.divergence.reason
    assert not ok.divergence.limits.calibrated
    high_kl = gate(_div(0.2, 0.99))
    assert high_kl.decision is Verdict.REVIEW
    assert "KL" in high_kl.divergence.reason and "uncalibrated" in high_kl.divergence.reason
    low_top1 = gate(_div(0.01, 0.9))
    assert low_top1.decision is Verdict.REVIEW
    assert "top-1" in low_top1.divergence.reason


def test_ceiling_must_lie_beyond_the_limits():
    with pytest.raises(ValueError, match="ceiling_kl"):
        GatePolicy(max_kl=0.5, ceiling_kl=0.5)
    with pytest.raises(ValueError, match="ceiling_top1"):
        GatePolicy(min_top1=0.8, ceiling_top1=0.9)


def test_review_ranks_between_pass_and_inconclusive():
    assert worst([Verdict.PASS, Verdict.REVIEW]) is Verdict.REVIEW
    assert worst([Verdict.REVIEW, Verdict.INCONCLUSIVE]) is Verdict.INCONCLUSIVE
    assert worst([Verdict.REVIEW, Verdict.FAIL]) is Verdict.FAIL


def test_sanity_failures():
    base = {"a": items(binary(1000, 0.8, 13))}
    clean = SanityResult(
        n=1000, counts={"empty": 0, "truncated": 5, "repetition": 0, "language_drift": 0}
    )
    assert evaluate_gate(base, base, None, clean, FAST).decision is Verdict.PASS
    loops = SanityResult(
        n=1000, counts={"empty": 0, "truncated": 5, "repetition": 80, "language_drift": 0}
    )
    d = evaluate_gate(base, base, None, loops, FAST)
    assert d.decision is Verdict.FAIL
    assert "repetition rate 8.00%" in d.sanity.reason
    strict = FAST.model_copy(update={"sanity": SanityLimits(max_truncated=0.001)})
    assert evaluate_gate(base, base, None, clean, strict).decision is Verdict.FAIL


def test_task_sets_must_match():
    with pytest.raises(ValueError, match="task sets differ"):
        evaluate_gate({"a": items([1.0] * 3)}, {"b": items([1.0] * 3)}, None, None, FAST)


def test_decision_is_json_able_for_store():
    base = {"a": items(binary(1000, 0.8, 14))}
    d = evaluate_gate(base, base, _div(0.2, 0.99), None, FAST, self_divergence=_floor(0.02, 0.98))
    doc = json.loads(json.dumps(d.details()))
    assert doc["decision"] == "review" and doc["blocked"] is False
    assert doc["tasks"][0]["task"] == "a"
    assert doc["policy"]["threshold"] == 0.01
    assert doc["policy"]["noise_multiple"] == 5.0 and doc["policy"]["review_blocks"] is False
    div = doc["divergence"]
    assert div["verdict"] == "review"
    assert div["result"]["kl"]["point"] == 0.2
    assert div["self_divergence"]["kl"]["hi"] == 0.02
    assert div["limits"]["calibrated"] is True
    assert div["limits"]["max_kl"] == pytest.approx(0.1)
    assert (div["limits"]["ceiling_kl"], div["limits"]["ceiling_top1"]) == (0.5, 0.8)
    assert "gate REVIEW" in d.summary()


def test_a_failed_divergence_is_inconclusive_and_keeps_the_task_verdicts():
    d = evaluate_gate(
        PASSING, PASSING, None, None, FAST, divergence_error="ValueError: no positions"
    )
    assert d.divergence.verdict is Verdict.INCONCLUSIVE
    assert d.divergence.reason == "not measured: ValueError: no positions"
    assert [t.verdict for t in d.tasks] == [Verdict.PASS]
    assert d.decision is Verdict.INCONCLUSIVE and d.blocked
