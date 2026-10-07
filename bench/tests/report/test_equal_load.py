import csv
import io

import pytest

from loom_bench.cost import CostAllocation
from loom_bench.records import LoadMode
from loom_bench.report.analyze import analyze_runs, default_price_resolver, load_points
from loom_bench.report.compare import compare
from loom_bench.report.compare import render_markdown as compare_markdown
from loom_bench.report.equal_load import (
    EQUAL_LOAD_METRICS,
    NO_SIGNIFICANT_DIFFERENCE,
    EqualLoadSide,
    disjoint_lower,
    equal_load,
    equal_load_for,
    shown_loads,
    side_of,
)
from loom_bench.report.leaderboard import (
    EQUAL_LOAD_CSV_COLUMNS,
    build_leaderboard,
    render_equal_load_csv,
    render_html,
    render_markdown,
)
from loom_bench.stats import Estimate

from .factories import SLO, TTFT_SGLANG, make_runs


def estimate(mean: float, lo: float | None, hi: float | None, n: int = 3) -> Estimate:
    return Estimate(mean=mean, lo=lo, hi=hi, n=n, std=None, method="log_t")


@pytest.fixture(scope="module")
def results(vllm_runs, sglang_runs, price_book):
    """vllm-bf16 (goodput 4 req/s) and sglang-bf16 (goodput 6), both at 2, 4, 6, 8."""
    return analyze_runs(
        [*vllm_runs, *sglang_runs],
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )


def test_disjoint_lower_needs_every_interval_to_clear():
    assert disjoint_lower([estimate(100, 90, 110), estimate(150, 140, 160)]) == 0
    assert disjoint_lower([estimate(150, 140, 160), estimate(100, 90, 110)]) == 1
    assert disjoint_lower([estimate(100, 90, 145), estimate(150, 140, 160)]) is None  # overlap
    # lower than one rival is not enough: it must clear every other config
    three = [estimate(100, 90, 110), estimate(150, 140, 160), estimate(105, 100, 112)]
    assert disjoint_lower(three) is None
    assert disjoint_lower([estimate(100, None, None, n=1), estimate(150, 140, 160)]) is None
    assert disjoint_lower([None, estimate(150, 140, 160)]) is None
    assert disjoint_lower([estimate(100, 90, 110)]) is None


def test_shown_loads_are_inside_the_slo_plus_the_highest_common_load(results):
    sides = [side_of(r) for r in results]
    # goodputs 4 and 6: every common load up to 4, then the highest common one
    assert shown_loads(sides) == [2.0, 4.0, 8.0]
    # without an SLO verdict (raw runs in compare): every common load
    raw = [EqualLoadSide(name=s.name, points=s.points) for s in sides]
    assert shown_loads(raw) == [2.0, 4.0, 6.0, 8.0]
    # loads only one side ran are never shown
    assert shown_loads([raw[0], raw[1].model_copy(update={"points": raw[1].points[1:]})]) == [
        4.0,
        6.0,
        8.0,
    ]


def test_equal_load_marks_only_significant_differences(results):
    table = equal_load_for(results)
    assert table is not None
    assert table.names == ["sglang-bf16", "vllm-bf16"]
    at = {p.load: p for p in table.points}
    # SGLang's TTFT is set 25% lower at 2 req/s, with 2% run-to-run jitter
    assert at[2.0].lower["ttft_ms.p95"] == "sglang-bf16"
    assert at[2.0].verdict("ttft_ms.p95") == "sglang-bf16 lower"
    # both decode at the same 30 ms per token: no winner
    assert at[2.0].lower["tpot_ms.p50"] is None
    assert at[2.0].verdict("tpot_ms.p50") == NO_SIGNIFICANT_DIFFERENCE
    # SLO verdicts at each load: 8 fails for both, 4 passes for both
    assert [c.slo_met for c in at[4.0].cells] == [True, True]
    assert [c.slo_met for c in at[8.0].cells] == [False, False]
    assert set(at[2.0].lower) == {m for m, _ in EQUAL_LOAD_METRICS}


def test_one_config_or_no_common_load_has_no_table(results, vllm_runs):
    assert equal_load_for(results[:1]) is None
    points = next(iter(load_points(vllm_runs).values()))
    a = EqualLoadSide(name="a", points=points[:1])
    b = EqualLoadSide(name="b", points=points[1:])
    assert equal_load([a, b], "chat", LoadMode.OPEN_LOOP) is None


def test_leaderboard_shows_the_equal_load_section(results):
    report = build_leaderboard(results)
    md = render_markdown(report)
    assert "### Latency at equal load" in md
    assert "| Load | Metric | sglang-bf16 | vllm-bf16 | Verdict |" in md
    assert "| 2 req/s | SLO at this load | met | met |  |" in md
    row = next(line for line in md.splitlines() if line.startswith("| 2 req/s | TTFT p95 |"))
    cells = [c.strip() for c in row.strip("|").split(" | ")]
    assert cells[2].startswith("**") and not cells[3].startswith("**")  # SGLang in bold
    assert cells[4] == "sglang-bf16 lower"
    html = render_html(report)
    assert "Latency at equal load" in html
    assert "<strong>" in html


def test_equal_load_csv_is_long_form(results):
    report = build_leaderboard(results)
    rows = list(csv.DictReader(io.StringIO(render_equal_load_csv(report))))
    assert list(rows[0]) == EQUAL_LOAD_CSV_COLUMNS
    assert len(rows) == 3 * 2 * len(EQUAL_LOAD_METRICS)  # loads x configs x metrics
    ttft = [r for r in rows if r["load"] == "2.0" and r["metric"] == "ttft_ms.p95"]
    flags = {r["config"]: r["significantly_lower"] for r in ttft}
    assert flags == {"sglang-bf16": "True", "vllm-bf16": "False"}
    assert all(r["verdict"] == "sglang-bf16 lower" for r in ttft)
    assert {r["value_ci_method"] for r in rows} == {"log_t"}


def test_single_config_board_has_no_equal_load_section(price_book):
    runs = make_runs("solo", engine="sglang", ttft=TTFT_SGLANG)
    results = analyze_runs(
        runs,
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )
    report = build_leaderboard(results)
    assert report.boards[0].equal_load is None
    assert "Latency at equal load" not in render_markdown(report)
    assert render_equal_load_csv(report).strip() == ",".join(EQUAL_LOAD_CSV_COLUMNS)


def test_compare_adds_equal_load_for_matched_sweeps(results, vllm_runs, sglang_runs):
    by = {r.name: r for r in results}
    c = compare([by["vllm-bf16"]], [by["sglang-bf16"]], match_by="workload")
    assert len(c.equal_load) == 1
    t = c.equal_load[0]
    assert t.names == ["A: vllm-bf16", "B: sglang-bf16"]
    assert [p.load for p in t.points] == [2.0, 4.0, 8.0]
    assert t.points[0].verdict("ttft_ms.p95") == "B: sglang-bf16 lower"
    md = compare_markdown(c)
    assert "## Latency at equal load" in md
    assert "| Load | Metric | A: vllm-bf16 | B: sglang-bf16 | Verdict |" in md

    # from raw runs there is no SLO verdict: every common load, SLO shown as n/a
    raw = compare(vllm_runs, sglang_runs, match_by="workload")
    assert [p.load for p in raw.equal_load[0].points] == [2.0, 4.0, 6.0, 8.0]
    assert "| 2 req/s | SLO at this load | n/a | n/a |  |" in compare_markdown(raw)
