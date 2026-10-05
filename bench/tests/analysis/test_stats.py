import math
from functools import partial

import numpy as np
import pytest
from scipy import stats as sps

from loom_bench.stats import (
    bootstrap_ci,
    log_mean_ci,
    mean_ci,
    paired_bootstrap_delta,
    percentile,
)


def test_percentile_linear_interpolation():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    # position (n-1)*q = 2.7 -> 3 + 0.7 * (4 - 3)
    assert percentile([4, 1, 3, 2], 90) == pytest.approx(3.7)
    assert percentile([7.0], 99) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_mean_ci_hand_computed():
    # mean 12, s = 2, t(0.975, df=2) = 4.302653, half = 4.302653 * 2 / sqrt(3)
    est = mean_ci([10, 12, 14])
    half = 4.302652729911275 * 2 / math.sqrt(3)
    assert est.mean == 12
    assert est.std == pytest.approx(2)
    assert est.lo == pytest.approx(12 - half)
    assert est.hi == pytest.approx(12 + half)
    assert est.n == 3
    assert est.trusted
    assert est.cv == pytest.approx(2 / 12)


@pytest.mark.parametrize("confidence", [0.9, 0.95, 0.99])
def test_mean_ci_matches_scipy(confidence):
    values = [3.1, 2.7, 3.9, 3.3, 2.95, 3.6]
    est = mean_ci(values, confidence)
    lo, hi = sps.t.interval(confidence, len(values) - 1, loc=np.mean(values), scale=sps.sem(values))
    assert est.lo == pytest.approx(lo)
    assert est.hi == pytest.approx(hi)
    assert est.confidence == confidence


def test_single_value_has_no_interval_and_is_untrusted():
    est = mean_ci([5.0])
    assert (est.mean, est.lo, est.hi, est.std, est.n) == (5.0, None, None, None, 1)
    assert not est.trusted
    assert est.cv is None
    assert est.model_dump()["trusted"] is False


def test_constant_values_have_zero_width_interval():
    est = mean_ci([2.0, 2.0, 2.0])
    assert est.lo == est.hi == est.mean == 2.0
    assert est.cv == 0.0


def test_mean_ci_rejects_bad_input():
    with pytest.raises(ValueError):
        mean_ci([])
    with pytest.raises(ValueError):
        mean_ci([1.0, float("nan")])
    with pytest.raises(ValueError):
        mean_ci([1.0, 2.0], confidence=1.0)


def test_bootstrap_is_deterministic_by_seed():
    rng = np.random.default_rng(1)
    values = rng.exponential(1.0, size=500)
    a = bootstrap_ci(values, n_boot=2000, seed=7)
    b = bootstrap_ci(values, n_boot=2000, seed=7)
    c = bootstrap_ci(values, n_boot=2000, seed=8)
    assert a == b
    assert (a.lo, a.hi) != (c.lo, c.hi)
    assert a.point == pytest.approx(values.mean())
    assert a.lo < a.point < a.hi


def test_bootstrap_percentile_statistic():
    values = np.arange(1, 201, dtype=float)
    ci = bootstrap_ci(values, partial(np.percentile, q=95), n_boot=1000, seed=0)
    assert ci.point == pytest.approx(np.percentile(values, 95))
    assert ci.lo <= ci.point <= ci.hi


def test_bootstrap_single_value_has_no_interval():
    ci = bootstrap_ci([3.0])
    assert ci.point == 3.0
    assert ci.lo is None and ci.hi is None


def test_bootstrap_chunking_does_not_change_coverage():
    # n large enough that resamples are drawn in several chunks.
    values = np.linspace(0, 1, 5000)
    ci = bootstrap_ci(values, n_boot=2000, seed=3)
    se = values.std(ddof=1) / math.sqrt(values.size)
    assert ci.hi - ci.lo == pytest.approx(2 * 1.96 * se, rel=0.1)


def test_paired_delta_constant_shift_is_exact():
    b = np.linspace(0, 1, 50)
    ci = paired_bootstrap_delta(b + 0.1, b, n_boot=500, seed=0)
    assert ci.point == pytest.approx(0.1)
    assert ci.lo == pytest.approx(0.1)
    assert ci.hi == pytest.approx(0.1)


def test_pairing_cancels_item_difficulty():
    rng = np.random.default_rng(0)
    difficulty = rng.uniform(0, 1, size=300)
    baseline = difficulty
    candidate = difficulty - 0.02 + rng.normal(0, 0.005, size=300)
    paired = paired_bootstrap_delta(candidate, baseline, n_boot=2000, seed=1)
    assert paired.hi < 0  # a 2-point drop is detected despite item spread
    unpaired_width = 2 * 1.96 * math.sqrt(2 * difficulty.var(ddof=1) / 300)
    assert paired.hi - paired.lo < unpaired_width / 10


def test_paired_delta_deterministic_and_validates_lengths():
    a, b = [1, 0, 1, 1, 0, 1], [1, 1, 0, 1, 0, 0]
    assert paired_bootstrap_delta(a, b, seed=4) == paired_bootstrap_delta(a, b, seed=4)
    with pytest.raises(ValueError):
        paired_bootstrap_delta([1, 2], [1, 2, 3])


def test_log_mean_ci_hand_computed():
    # ln 100 = 4.605170, ln 400 = 5.991465: m = 5.298317 (= ln 200), s_log = ln 4 / sqrt 2
    # t(0.975, df=1) = 12.706205; half = 12.706205 * (ln 4 / sqrt 2) / sqrt 2 = 6.353102 * ln 4
    est = log_mean_ci([100.0, 400.0])
    half = 12.706204736174698 * math.log(4) / 2
    assert est.method == "log_t"
    assert est.mean == pytest.approx(200.0)  # geometric mean, not 250
    assert est.lo == pytest.approx(200.0 * math.exp(-half))
    assert est.hi == pytest.approx(200.0 * math.exp(half))
    assert est.lo * est.hi == pytest.approx(est.mean**2)  # multiplicative: GM ×/÷ factor
    assert est.log_std == pytest.approx(math.log(4) / math.sqrt(2))
    assert est.std == pytest.approx(math.sqrt(2) * 150)  # raw-scale spread is kept
    assert est.cv == pytest.approx(math.sqrt(math.expm1(est.log_std**2)))


def test_log_mean_ci_matches_scipy_on_logs():
    values = [3.1, 2.7, 3.9, 3.3, 2.95, 3.6]
    est = log_mean_ci(values, 0.9)
    logs = np.log(values)
    lo, hi = sps.t.interval(0.9, len(values) - 1, loc=logs.mean(), scale=sps.sem(logs))
    assert (est.lo, est.hi) == (pytest.approx(math.exp(lo)), pytest.approx(math.exp(hi)))


@pytest.mark.parametrize(
    "values",
    [
        [2.9, 41.7],  # one fast and one slow repetition: arithmetic lo is about -230
        [1152.0 * 0.6, 1152.0 * 1.4],
        [5e-3, 4.0, 0.2],
    ],
)
def test_log_interval_bounds_stay_positive_where_t_goes_negative(values):
    assert mean_ci(values).lo < 0
    est = log_mean_ci(values)
    assert 0 < est.lo <= est.mean <= est.hi


def test_log_mean_ci_rejects_non_positive_and_single_value_has_no_interval():
    with pytest.raises(ValueError, match="strictly positive"):
        log_mean_ci([1.0, 0.0])
    one = log_mean_ci([7.0])
    assert (one.mean, one.lo, one.hi, one.method, one.trusted) == (7.0, None, None, "log_t", False)


def test_clipped_t_interval():
    est = mean_ci([0.0, 0.02], lower=0.0, upper=1.0)
    assert est.method == "t_clipped"
    assert est.lo == 0.0 and est.hi == pytest.approx(0.01 + 12.706204736174698 * 0.01)
    full = mean_ci([0.9, 1.0, 1.0], lower=0.0, upper=1.0)
    assert full.hi == 1.0
    assert mean_ci([1.0, 2.0]).method == "t"


def test_scaled_keeps_method_and_relative_width():
    est = log_mean_ci([100.0, 120.0, 90.0]).scaled(0.5)
    base = log_mean_ci([50.0, 60.0, 45.0])
    assert est.method == "log_t"
    assert (est.mean, est.lo, est.hi) == (
        pytest.approx(base.mean),
        pytest.approx(base.lo),
        pytest.approx(base.hi),
    )
    assert est.cv == pytest.approx(base.cv)
    with pytest.raises(ValueError):
        est.scaled(0)


def test_log_mean_ci_of_identical_values_is_exact():
    est = log_mean_ci([0.3, 0.3, 0.3])
    assert est.mean == est.lo == est.hi == 0.3
    assert est.cv == 0.0
