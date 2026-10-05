import json

import numpy as np
import pytest

from loom_bench.quality.divergence import DivergenceResult
from loom_bench.quality.gate import GatePolicy, TaskPolicy, Verdict, evaluate_gate, evaluate_task
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


def test_divergence_failures():
    base = {"a": items(binary(1000, 0.8, 12))}
    ok = evaluate_gate(base, base, _div(0.01, 0.99), None, FAST)
    assert ok.decision is Verdict.PASS and not ok.blocked
    high_kl = evaluate_gate(base, base, _div(0.2, 0.99), None, FAST)
    assert high_kl.decision is Verdict.FAIL
    assert "KL" in high_kl.divergence.reason
    low_top1 = evaluate_gate(base, base, _div(0.01, 0.9), None, FAST)
    assert low_top1.decision is Verdict.FAIL
    assert "top-1" in low_top1.divergence.reason


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
    d = evaluate_gate(base, base, _div(0.01, 0.99), None, FAST)
    doc = json.loads(json.dumps(d.details()))
    assert doc["decision"] == "pass"
    assert doc["tasks"][0]["task"] == "a"
    assert doc["policy"]["threshold"] == 0.01
    assert "gate PASS" in d.summary()
