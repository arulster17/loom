import csv
import io
import math

import pytest

from loom_bench.cost import CostAllocation
from loom_bench.records import LoadMode
from loom_bench.report.analyze import analyze_runs, default_price_resolver
from loom_bench.report.compare import compare
from loom_bench.report.compare import render_markdown as compare_markdown
from loom_bench.report.format import goodput_bracket, load_num
from loom_bench.report.leaderboard import build_leaderboard, render_csv, render_markdown
from loom_bench.report.resolution import brackets_overlap, goodput_bounds, goodput_ties
from loom_bench.slo import GoodputResult

from .factories import SLO, TTFT_SGLANG, TTFT_VLLM, make_runs


def goodput(
    max_load: float | None,
    first_fail: float | None,
    mode: LoadMode = LoadMode.OPEN_LOOP,
) -> GoodputResult:
    return GoodputResult(
        load_mode=mode,
        slo=SLO,
        max_load=max_load,
        first_failing_load=first_fail,
        bracketed=first_fail is not None,
        trusted=True,
        request_rate=None,
        output_tok_s=None,
        input_tok_s=None,
        total_tok_s=None,
        max_sustainable_concurrency=None,
        points=[],
    )


def test_bracket_text_covers_every_case():
    assert goodput_bracket(goodput(1.0, 1.0905077)) == "1 req/s (fails at 1.091)"
    assert goodput_bracket(goodput(8.0, None)) == "≥8 req/s (none failed)"
    assert goodput_bracket(goodput(None, 0.5)) == "none met the SLO (fails at 0.5)"
    assert goodput_bracket(goodput(16, 32, LoadMode.CLOSED_LOOP)) == ("16 concurrent (fails at 32)")
    assert load_num(1.2968395) == "1.297"
    assert load_num(6.0) == "6"


def test_bounds_are_half_open_and_unbounded_without_a_failure():
    assert goodput_bounds(goodput(1.0, 1.09)) == (1.0, 1.09)
    assert goodput_bounds(goodput(8.0, None)) == (8.0, math.inf)
    assert goodput_bounds(goodput(None, 0.5)) is None


@pytest.mark.parametrize(
    ("a", "b", "tie"),
    [
        ((1.0, 1.09), (1.0, 1.09), True),  # the same grid point: the 8B fixed-1k-1k case
        ((1.0, 1.19), (1.09, 1.19), True),  # nested brackets
        ((1.30, 1.41), (1.09, 1.19), False),  # the 8B code-completion case: separated
        ((4.0, 6.0), (6.0, 8.0), False),  # touching: 6 failed for one, passed for the other
        ((8.0, None), (8.0, 10.0), True),  # a lower bound overlapping a bracket
        ((12.0, None), (8.0, 10.0), False),
        ((None, 0.5), (None, 0.5), False),  # no goodput on either side: nothing to tie
        ((None, 0.5), (0.25, 0.5), False),
    ],
)
def test_brackets_overlap(a, b, tie):
    assert brackets_overlap(goodput(*a), goodput(*b)) is tie
    assert brackets_overlap(goodput(*b), goodput(*a)) is tie


@pytest.fixture(scope="module")
def tied(price_book):
    """Two engines with the same latency on the same grid: identical brackets."""
    runs = [
        *make_runs("vllm-a", ttft=TTFT_VLLM),
        *make_runs("sglang-b", engine="sglang", ttft=TTFT_VLLM),
        *make_runs("sglang-c", engine="sglang", ttft=TTFT_SGLANG),
    ]
    return analyze_runs(
        runs,
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )


def test_ties_are_marked_on_the_board_and_in_the_recommendation(tied):
    by = {r.name: r for r in tied}
    assert [t.name for t in goodput_ties(by["vllm-a"], tied)] == ["sglang-b"]
    assert goodput_ties(by["sglang-c"], tied) == []

    rows = {row.result.name: row for row in build_leaderboard(tied).boards[0].rows}
    assert rows["sglang-c"].recommendation.startswith("Cheapest at SLO; 33% cheaper than")
    assert rows["vllm-a"].goodput_ties == ["sglang-b"]
    assert rows["vllm-a"].goodput_text == "4 req/s (fails at 6); tied with sglang-b"
    tied_rows = [r for r in rows.values() if r.goodput_ties]
    assert {r.result.name for r in tied_rows} == {"vllm-a", "sglang-b"}
    # neither is tied with the leader, so both compare to it and name their tie
    for row in tied_rows:
        other = ({"vllm-a", "sglang-b"} - {row.result.name}).pop()
        assert "more expensive than sglang-c at SLO" in row.recommendation
        assert f"tied with {other} within the search resolution" in row.recommendation


def test_leader_tie_says_tied_for_cheapest(price_book):
    runs = [
        *make_runs("vllm-a", ttft=TTFT_VLLM),
        *make_runs("sglang-b", engine="sglang", ttft=TTFT_VLLM),
    ]
    results = analyze_runs(
        runs,
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )
    rows = build_leaderboard(results).boards[0].rows
    assert rows[0].recommendation.startswith(
        f"Tied for cheapest at SLO with {rows[1].result.name}: goodput brackets overlap"
    )
    assert rows[1].recommendation.startswith(f"Tied with {rows[0].result.name} at SLO")


def test_bracket_columns_in_markdown_and_csv(tied):
    report = build_leaderboard(tied)
    md = render_markdown(report)
    assert "Goodput load (search bracket)" in md
    assert "tied within the search resolution" in md
    rows = {r["config"]: r for r in csv.DictReader(io.StringIO(render_csv(report)))}
    assert rows["vllm-a"]["goodput_bracket"] == "4 req/s (fails at 6)"
    assert rows["vllm-a"]["goodput_at_least"] == "False"
    assert rows["vllm-a"]["goodput_ties"] == "sglang-b"
    assert rows["sglang-c"]["goodput_ties"] == ""


def test_compare_flags_a_tie_within_the_search_resolution(tied):
    by = {r.name: r for r in tied}
    same = compare([by["vllm-a"]], [by["sglang-b"]], match_by="workload")
    cfg = same.configs[0]
    assert cfg.goodput_tie
    assert cfg.goodput_bracket_a == cfg.goodput_bracket_b == "4 req/s (fails at 6)"
    assert "a tie within the search resolution" in compare_markdown(same)

    apart = compare([by["vllm-a"]], [by["sglang-c"]], match_by="workload")
    assert not apart.configs[0].goodput_tie
    assert "tie within the search resolution" not in compare_markdown(apart)
