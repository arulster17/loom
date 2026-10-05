import pytest

from loom_bench.metrics.gpu import NVIDIA_SMI_ARGS, parse_nvidia_smi, summarize_gpu

SMI = """\
2026/10/04 12:00:00.000, 0, 80, 30000, 46068, 250.50
2026/10/04 12:00:00.000, 1, 60, 20000, 46068, 200.00

2026/10/04 12:00:01.000, 0, 100, 31000, 46068, 300.00
2026/10/04 12:00:01.000, 1, [N/A], 21000, 46068, [N/A]
"""


def test_query_args_match_parser_columns():
    assert NVIDIA_SMI_ARGS[1] == (
        "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw"
    )


def test_parse_nvidia_smi():
    samples = parse_nvidia_smi(SMI)
    assert len(samples) == 4
    s = samples[3]
    assert s.index == 1
    assert s.utilization_pct is None
    assert s.power_w is None
    assert s.memory_used_mib == 21000


def test_summarize_gpu_per_gpu_and_overall():
    m = summarize_gpu(parse_nvidia_smi(SMI))
    assert m.n_gpus == 2
    g0, g1 = m.gpus
    assert g0.index == 0
    assert g0.utilization_mean_pct == 90
    assert g0.utilization_p95_pct == pytest.approx(99)  # 80 + 0.95 * 20
    assert g0.memory_peak_mib == 31000
    assert g0.power_mean_w == pytest.approx(275.25)
    assert g1.utilization_mean_pct == 60
    assert g1.power_mean_w == 200
    assert g1.memory_peak_mib == 21000

    assert m.overall.n_samples == 4
    assert m.overall.utilization_mean_pct == pytest.approx(80)
    assert m.overall.utilization_p95_pct == pytest.approx(98)  # [60, 80, 100] at 95%
    assert m.overall.memory_peak_mib == 31000
    assert m.overall.memory_total_mib == 46068
    assert m.overall.power_mean_w == pytest.approx((250.5 + 200 + 300) / 3)


def test_empty_and_malformed():
    m = summarize_gpu(parse_nvidia_smi(""))
    assert m.n_gpus == 0
    assert m.overall.utilization_mean_pct is None
    with pytest.raises(ValueError):
        parse_nvidia_smi("2026/10/04 12:00:00.000, 0, 80")
    with pytest.raises(ValueError):
        parse_nvidia_smi("2026/10/04 12:00:00.000, 0, busy, 1, 2, 3")
