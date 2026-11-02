"""The leaderboard's speculative-decoding table and CSV columns."""

import csv
import io

import pytest

from loom_bench.cost import CostAllocation
from loom_bench.metrics.prometheus import ServerMetrics
from loom_bench.report.analyze import analyze_runs, default_price_resolver
from loom_bench.report.leaderboard import (
    build_leaderboard,
    render_csv,
    render_html,
    render_markdown,
)
from loom_bench.report.specdec import specdec_lines, specdec_rows, specdec_table

from .factories import SLO, TPOT_MS, make_runs

EAGLE = {
    "max_num_seqs": 256,
    "speculative_config": {
        "method": "eagle3",
        "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "revision": "08610ffa01dd9f16731fe8f627b85905b6aa51c4",
        "num_speculative_tokens": 3,
    },
}
# TPOT p50 as a share of the reference's, and accepted tokens per step, by load: the
# gain shrinks as the batch grows and turns into a loss past 4 req/s.
TPOT_SHARE = {2.0: 0.5, 4.0: 0.8, 6.0: 1.1, 8.0: 1.3}
ACCEPTED = {2.0: 2.4, 4.0: 2.3, 6.0: 2.2, 8.0: 2.1}


def _speculating(runs):
    """Runs of a config whose server counters show speculation, TPOT p50 per TPOT_SHARE."""
    for run in runs:
        load = run.load_value
        s = run.summary
        s["tpot_ms"]["p50"] = TPOT_MS * TPOT_SHARE[load] * (1 + 0.01 * run.repetition)
        length = ACCEPTED[load] + 0.01 * run.repetition
        s["server"] = ServerMetrics(
            engine="vllm",
            n_scrapes=10,
            duration_s=9.0,
            running_mean=None,
            running_max=None,
            waiting_mean=None,
            waiting_max=None,
            kv_usage_mean=None,
            kv_usage_max=None,
            preemptions=None,
            prefix_cache_hit_rate=None,
            queue_time_mean_ms=None,
            missing=[],
            spec_drafts=1000,
            spec_draft_tokens=3000,
            spec_accepted_tokens=(length - 1) * 1000,
            spec_acceptance_rate=(length - 1) / 3,
            spec_mean_accepted_length=length,
            spec_acceptance_by_position={"0": 0.7, "1": 0.45, "2": (length - 1) - 1.15},
        ).model_dump(mode="json")
    return runs


@pytest.fixture(scope="module")
def report(vllm_runs, price_book):
    spec = _speculating(make_runs("eagle3", engine_args=EAGLE))
    results = analyze_runs(
        [*vllm_runs, *spec],
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )
    return build_leaderboard(results)


def test_a_board_without_speculation_has_no_table(vllm_runs, sglang_runs, price_book):
    results = analyze_runs(
        [*vllm_runs, *sglang_runs],
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )
    assert specdec_table(results) is None
    board = build_leaderboard(results)
    assert all(b.specdec is None for b in board.boards)
    assert "Speculative decoding" not in render_markdown(board)
    assert "Speculative decoding" not in render_html(board)


def test_rows_per_load_with_the_reference_and_the_crossover(report):
    (board,) = report.boards
    t = board.specdec
    assert t is not None and t.reference == "vllm-bf16"
    assert [(r.load, r.config) for r in t.rows][:4] == [
        (2.0, "vllm-bf16"),
        (2.0, "eagle3"),
        (4.0, "vllm-bf16"),
        (4.0, "eagle3"),
    ]
    by = {(r.load, r.config): r for r in t.rows}
    assert by[(2.0, "eagle3")].accepted_length.mean == pytest.approx(2.42, abs=0.01)
    assert by[(2.0, "eagle3")].speedup == pytest.approx(2.0, rel=0.03)
    assert by[(8.0, "eagle3")].speedup == pytest.approx(1 / 1.3, rel=0.03)
    assert by[(2.0, "vllm-bf16")].accepted_length is None
    assert by[(2.0, "vllm-bf16")].speedup == pytest.approx(1.0)
    assert len(by[(2.0, "eagle3")].by_position) == 3
    assert "stops helping between 4 req/s and 6 req/s" in t.crossover["eagle3"]
    # Repetition jitter differs a little between the two configs: 1.96x, not 2.
    assert t.crossover["eagle3"].startswith("TPOT p50 speed-up 1.96x at 2 req/s, 1.23x at 4")
    lines = specdec_lines(t)
    assert lines[0] == "Reference (no speculation): vllm-bf16."
    cells = specdec_rows(t)[1]
    assert cells[0] == "2 req/s" and cells[1] == "eagle3"
    assert cells[2].startswith("2.42 [") and cells[8] == "1.96x"
    assert "(0.70, 0.45, 0.27)" in cells[3]


def test_markdown_and_html_show_the_table(report):
    md = render_markdown(report)
    assert "#### Speculative decoding by load" in md
    assert "| Load | Config | Accepted per step |" in md
    assert "- eagle3: TPOT p50 speed-up" in md
    html = render_html(report)
    assert "<h4>Speculative decoding by load</h4>" in html
    assert "stops helping between 4 req/s and 6 req/s" in html


def test_csv_reports_acceptance_at_goodput(report):
    rows = {r["config"]: r for r in csv.DictReader(io.StringIO(render_csv(report)))}
    spec, plain = rows["eagle3"], rows["vllm-bf16"]
    assert plain["spec_mean_accepted_length"] == "" and plain["spec_acceptance_rate"] == ""
    goodput = float(spec["goodput_load"])
    assert float(spec["spec_mean_accepted_length"]) == pytest.approx(
        ACCEPTED[goodput] + 0.02, abs=0.005
    )
    assert float(spec["spec_acceptance_rate"]) == pytest.approx(
        (ACCEPTED[goodput] + 0.02 - 1) / 3, abs=0.005
    )
    assert spec["spec_acceptance_rate_ci_method"] == "t_clipped"


def test_no_single_reference_means_no_speed_up(vllm_runs, sglang_runs, price_book):
    spec = _speculating(make_runs("eagle3", engine_args=EAGLE))
    results = analyze_runs(
        [*vllm_runs, *sglang_runs, *spec],
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )
    t = specdec_table(results)
    assert t is not None and t.reference is None and t.crossover == {}
    assert all(r.speedup is None for r in t.rows)
    assert specdec_lines(t)[0].startswith("No single config without speculation")
