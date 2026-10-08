"""Replicated eval passes: pooling into per-item means, and what it does to the gate.

The numbers mirror the Qwen3-8B IFEval measurements on vLLM: of 541 items, 32 changed
score across three independent runs (greedy decoding is not deterministic under
batching), so two runs of the *same* engine disagree on about 21 items and a single
paired comparison reads a CI of about +-1.5 pts out of pure noise.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

import pytest

from loom_bench.quality.gate import GATE_METHOD, GatePolicy, TaskPolicy, Verdict, evaluate_gate
from loom_bench.quality.runner import SuiteResult, TaskRun, pool_replicates, score_estimate
from loom_bench.quality.sanity import SanityResult
from loom_bench.quality.tasks.base import ItemResult

N = 541
NOISY = 32  # items whose outcome flips between runs of one engine
MARGIN = 0.02  # IFEval's non-inferiority margin in the Qwen3-8B suite
T0 = datetime(2026, 10, 8, tzinfo=UTC)


def _suite(scores: dict[str, float], *, version: str = "1", sanity_n: int = 10) -> SuiteResult:
    items = [ItemResult(item_id=k, score=v, content_hash=f"h-{k}") for k, v in scores.items()]
    run = TaskRun(
        name="ifeval", kind="lm_eval", version=version, items=items, estimate=score_estimate(items)
    )
    return SuiteResult(
        suite="s",
        model="m",
        tasks={"ifeval": run},
        sanity=SanityResult(n=sanity_n, counts={"empty": 1, "truncated": 0}),
        started_at=T0,
        finished_at=T0,
    )


def _probs(shift_items: int = 0, seed: int = 7) -> list[float]:
    """Per-item pass probabilities: 82% of items always pass, the rest always fail,
    and NOISY items are coin flips. `shift_items` always-pass items become always-fail
    (a genuine regression of shift_items / N)."""
    rng = random.Random(seed)
    p = [1.0 if rng.random() < 0.82 else 0.0 for _ in range(N)]
    for i in rng.sample(range(N), NOISY):
        p[i] = 0.5
    always = [i for i in range(N) if p[i] == 1.0]
    for i in always[:shift_items]:
        p[i] = 0.0
    return p


def _passes(p: list[float], r: int, rng: random.Random) -> SuiteResult:
    runs = [_suite({f"i{i}": float(rng.random() < pi) for i, pi in enumerate(p)}) for _ in range(r)]
    return pool_replicates(runs)


def _gate(base: SuiteResult, cand: SuiteResult):
    policy = GatePolicy(tasks={"ifeval": TaskPolicy(threshold=MARGIN, min_samples=500)})
    return evaluate_gate(
        base.scores(),
        cand.scores(),
        None,
        None,
        policy,
        baseline_replicates=base.replicates,
        candidate_replicates=cand.replicates,
    )


def test_pooling_averages_items_and_keeps_every_pass():
    a = _suite({"x": 1.0, "y": 0.0}, sanity_n=10)
    b = _suite({"x": 0.0, "y": 0.0}, sanity_n=10)
    c = _suite({"x": 1.0, "y": 0.0}, sanity_n=10)
    pooled = pool_replicates([a, b, c])
    assert pooled.replicates == 3
    items = {i.item_id: i for i in pooled.tasks["ifeval"].items}
    assert items["x"].score == pytest.approx(2 / 3)
    assert items["x"].meta["replicate_scores"] == [1.0, 0.0, 1.0]
    assert items["x"].content_hash == "h-x"
    assert pooled.tasks["ifeval"].estimate.mean == pytest.approx(1 / 3)
    assert pooled.tasks["ifeval"].provenance["replicate_means"] == [0.5, 0.0, 0.5]
    assert pooled.sanity.n == 30 and pooled.sanity.counts["empty"] == 3


def test_one_pass_pools_to_itself():
    a = _suite({"x": 1.0})
    assert pool_replicates([a]) is a


@pytest.mark.parametrize(
    ("other", "match"),
    [
        (_suite({"x": 1.0}, version="2"), "versions"),
        (_suite({"z": 1.0}), "different items"),
    ],
)
def test_pooling_refuses_passes_that_do_not_match(other, match):
    with pytest.raises(ValueError, match=match):
        pool_replicates([_suite({"x": 1.0}), other])


def test_replicates_narrow_the_ci_from_engine_noise():
    # Same engine on both sides. One pass per side is as wide as 565b8d3f's reading;
    # three passes cut the width by roughly sqrt(3) and the same config passes.
    p = _probs()
    widths = {}
    for r in (1, 3):
        ws = []
        for seed in range(20):
            rng = random.Random(seed)
            v = _gate(_passes(p, r, rng), _passes(p, r, rng)).tasks[0]
            assert v.ci_high is not None and v.ci_low is not None
            ws.append(v.ci_high - v.ci_low)
        widths[r] = sum(ws) / len(ws)
    assert widths[3] < 0.7 * widths[1]
    rng = random.Random(99)
    decision = _gate(_passes(p, 3, rng), _passes(p, 3, rng))
    assert decision.tasks[0].verdict is Verdict.PASS
    assert decision.method == GATE_METHOD
    assert (decision.baseline_replicates, decision.candidate_replicates) == (3, 3)


@pytest.mark.parametrize("replicates", [1, 3])
def test_a_genuine_five_point_regression_still_fails(replicates):
    # 27 always-pass items always fail on the candidate (-5 pts), under the same noise:
    # averaging the noise must not average the regression away.
    base_p, cand_p = _probs(), _probs(shift_items=27)
    for seed in range(10):
        rng = random.Random(seed)
        v = _gate(_passes(base_p, replicates, rng), _passes(cand_p, replicates, rng)).tasks[0]
        assert v.verdict is Verdict.FAIL, v.reason


def test_a_real_drop_inside_the_ci_is_named_not_hidden():
    # -1.5 pts that is really there: the CI excludes 0 but reaches past the -2 pt
    # margin, so the verdict is INCONCLUSIVE and the reason says the drop is real.
    base = _suite({f"i{i}": 1.0 for i in range(N)})
    cand = _suite({f"i{i}": 0.0 if i < 8 else 1.0 for i in range(N)})
    v = _gate(base, cand).tasks[0]
    assert v.verdict is Verdict.INCONCLUSIVE
    assert v.ci_high is not None and v.ci_high < 0
    assert "a real drop (CI below 0)" in v.reason and "more samples needed" in v.reason


def test_a_small_real_drop_within_the_margin_passes_and_says_so():
    n = 3000
    base = _suite({f"i{i}": 1.0 for i in range(n)})
    cand = _suite({f"i{i}": 0.0 if i < 15 else 1.0 for i in range(n)})  # -0.5 pts
    policy = GatePolicy(tasks={"ifeval": TaskPolicy(threshold=MARGIN, min_samples=500)})
    v = evaluate_gate(base.scores(), cand.scores(), None, None, policy).tasks[0]
    assert v.verdict is Verdict.PASS
    assert v.reason.endswith("but measurably lower (CI below 0)")
