"""Eval task scores are fractions in [0, 1], so their CIs are clipped to that range."""

from datetime import UTC, datetime

import pytest

from loom_bench.quality.runner import SuiteResult, TaskRun, score_estimate
from loom_bench.quality.sanity import SanityResult
from loom_bench.quality.tasks.base import ItemResult
from loom_bench.runner import read_samples, write_samples
from loom_bench.stats import mean_ci


def items(ones: int, zeros: int) -> list[ItemResult]:
    scores = [1.0] * ones + [0.0] * zeros
    return [ItemResult(item_id=str(i), score=s) for i, s in enumerate(scores)]


def test_a_near_perfect_score_stays_inside_one():
    # vLLM tool_calling in 565b8d3f: 58 of 60, reported as 0.967 [0.920, 1.013].
    scored = items(58, 2)
    plain = mean_ci([i.score for i in scored])
    assert plain.hi is not None and plain.hi > 1.0
    est = score_estimate(scored)
    assert est.mean == pytest.approx(58 / 60)
    assert est.method == "t_clipped"
    assert est.lo == pytest.approx(plain.lo) and est.hi == 1.0


def test_a_near_zero_score_stays_above_zero():
    est = score_estimate(items(1, 59))
    assert est.lo == 0.0 and est.hi is not None and 0.0 < est.hi < 1.0


def test_a_mid_range_score_is_the_plain_interval():
    scored = items(30, 30)
    plain, est = mean_ci([i.score for i in scored]), score_estimate(scored)
    assert (est.lo, est.hi) == (plain.lo, plain.hi)


def test_stored_samples_reload_with_clipped_intervals(tmp_path):
    scored = items(58, 2)
    now = datetime.now(UTC)
    result = SuiteResult(
        suite="s",
        model="m",
        tasks={"t": TaskRun("t", "tool_calling", "1", scored, score_estimate(scored))},
        sanity=SanityResult(n=0, counts={}),
        started_at=now,
        finished_at=now,
    )
    _, back = read_samples(write_samples(tmp_path / "samples.json", "s", result))
    assert back.tasks["t"].estimate.hi == 1.0
