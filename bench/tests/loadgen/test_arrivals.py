import numpy as np
import pytest
from pydantic import ValidationError

from loom_bench.loadgen import arrivals as arr


def rng(seed=0):
    return np.random.default_rng(seed)


def check_schedule(t, duration):
    assert t.dtype == np.float64
    assert np.all(np.diff(t) >= 0)
    assert t.size == 0 or (t[0] >= 0 and t[-1] < duration)


def test_constant():
    t = arr.constant(4.0, 2.0)
    np.testing.assert_allclose(t, np.arange(8) / 4.0)
    check_schedule(t, 2.0)


def test_poisson_mean_rate_and_exponential_gaps():
    t = arr.poisson(50.0, 400.0, rng())
    check_schedule(t, 400.0)
    assert t.size / 400.0 == pytest.approx(50.0, rel=0.02)
    gaps = np.diff(t)
    assert gaps.std() / gaps.mean() == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("burstiness", [0.25, 1.0, 4.0])
def test_gamma_rate_and_cv(burstiness):
    t = arr.gamma(20.0, burstiness, 1000.0, rng(1))
    check_schedule(t, 1000.0)
    assert t.size / 1000.0 == pytest.approx(20.0, rel=0.05)
    gaps = np.diff(t)
    assert gaps.std() / gaps.mean() == pytest.approx(1 / np.sqrt(burstiness), rel=0.05)


def test_onoff_burst_rates_per_phase():
    t = arr.onoff_burst(40.0, 2.0, t_on_s=5.0, t_off_s=15.0, duration_s=2000.0, rng=rng())
    check_schedule(t, 2000.0)
    on = np.mod(t, 20.0) < 5.0
    assert on.sum() / (2000.0 * 5 / 20) == pytest.approx(40.0, rel=0.05)
    assert (~on).sum() / (2000.0 * 15 / 20) == pytest.approx(2.0, rel=0.1)


def test_onoff_with_silent_off_phase():
    t = arr.onoff_burst(10.0, 0.0, 1.0, 1.0, 100.0, rng())
    assert np.all(np.mod(t, 2.0) < 1.0)


def test_ramp_up_and_down_shape():
    up = arr.ramp(0.0, 40.0, 1000.0, rng())
    check_schedule(up, 1000.0)
    # Linear 0->40: total 20k; first half holds 1/4 of the arrivals.
    assert up.size == pytest.approx(20_000, rel=0.03)
    assert (up < 500).sum() / up.size == pytest.approx(0.25, abs=0.02)
    down = arr.ramp(40.0, 0.0, 1000.0, rng())
    assert (down < 500).sum() / down.size == pytest.approx(0.75, abs=0.02)


def test_diurnal_peak_and_trough():
    period = 100.0
    t = arr.diurnal(20.0, 0.8, period, 2000.0, rng())
    check_schedule(t, 2000.0)
    assert t.size / 2000.0 == pytest.approx(20.0, rel=0.03)
    phase = np.mod(t, period) / period
    trough = ((phase < 0.1) | (phase >= 0.9)).sum()
    peak = ((phase >= 0.4) & (phase < 0.6)).sum()
    # Integrated rate over each 20% window: 1 -/+ 0.8*sin(0.2*pi)/(0.2*pi).
    k = 0.8 * np.sin(0.2 * np.pi) / (0.2 * np.pi)
    assert peak / trough == pytest.approx((1 + k) / (1 - k), rel=0.1)


def test_schedules_are_deterministic_by_seed():
    for spec in [
        {"kind": "poisson", "rate": 5},
        {"kind": "gamma", "rate": 5, "burstiness": 0.5},
        {"kind": "onoff_burst", "rate_hi": 9, "rate_lo": 1, "t_on_s": 2, "t_off_s": 3},
        {"kind": "ramp", "rate_start": 1, "rate_end": 9},
        {"kind": "diurnal", "mean_rate": 5, "amplitude": 0.5, "period_s": 30},
    ]:
        s = arr.parse_arrivals(spec)
        a, b, c = s.schedule(60, seed=7), s.schedule(60, seed=7), s.schedule(60, seed=8)
        np.testing.assert_array_equal(a, b)
        assert a.size != c.size or not np.array_equal(a, c)
        check_schedule(a, 60)


def test_spec_validation():
    assert isinstance(arr.parse_arrivals({"kind": "constant", "rate": 2}), arr.ConstantArrivals)
    with pytest.raises(ValidationError):
        arr.parse_arrivals({"kind": "poisson", "rate": 0})
    with pytest.raises(ValidationError):
        arr.parse_arrivals({"kind": "poisson", "rate": 1, "burst": 2})
    with pytest.raises(ValidationError):
        arr.parse_arrivals({"kind": "ramp", "rate_start": 0, "rate_end": 0})
    with pytest.raises(ValidationError):
        arr.parse_arrivals({"kind": "diurnal", "mean_rate": 1, "amplitude": 1.5, "period_s": 1})
    with pytest.raises(ValidationError):
        arr.parse_arrivals({"kind": "nope"})


AZURE = """TIMESTAMP,ContextTokens,GeneratedTokens
2023-11-16 18:15:46.6805900,4808,10
2023-11-16 18:15:50.9951690,3180,8
2023-11-16 18:15:47.1805900,110,27
2023-11-16 18:15:56.6805900,7433,14
"""

BURSTGPT = """Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type
5,ChatGPT,472,18,490,Conversation log
45,ChatGPT,1087,0,1087,Conversation log
110.5,GPT-4,417,505,922,API log
"""


def test_azure_trace_parse_sort_and_scale(tmp_path):
    p = tmp_path / "azure.csv"
    p.write_text(AZURE)
    rows = arr.read_azure_trace(p)
    assert [r.input_tokens for r in rows] == [4808, 110, 3180, 7433]
    assert [r.output_tokens for r in rows] == [10, 27, 8, 14]
    assert [r.timestamp_s for r in rows] == pytest.approx([0.0, 0.5, 4.314579, 10.0])
    np.testing.assert_allclose(arr.trace_offsets(rows, time_scale=0.5), [0, 0.25, 2.1572895, 5])
    np.testing.assert_allclose(arr.trace_offsets(rows, duration_s=5.0), [0, 0.5, 4.314579])
    assert len(arr.read_azure_trace(p, max_rows=2)) == 2


def test_burstgpt_trace_and_spec(tmp_path, monkeypatch):
    p = tmp_path / "burst.csv"
    p.write_text(BURSTGPT)
    rows = arr.read_burstgpt_trace(p)
    assert [(r.timestamp_s, r.input_tokens, r.output_tokens) for r in rows] == [
        (0.0, 472, 18),
        (40.0, 1087, 0),
        (105.5, 417, 505),
    ]
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    spec = arr.parse_arrivals(
        {"kind": "trace", "path": "$TRACE_DIR/burst.csv", "format": "burstgpt", "time_scale": 0.1}
    )
    np.testing.assert_allclose(spec.schedule(duration_s=10.0, seed=0), [0.0, 4.0])


def test_trace_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing trace columns"):
        arr.read_azure_trace(p)
