import csv
import io

import pytest

from loom_bench.cost import CostAllocation, PriceColumn
from loom_bench.report.analyze import (
    ColdStartStat,
    analyze_runs,
    default_price_resolver,
    with_quality,
)
from loom_bench.report.format import UNBRACKETED_NOTE
from loom_bench.report.leaderboard import (
    CSV_COLUMNS,
    RowStatus,
    build_leaderboard,
    cold_text,
    md_headers,
    render_csv,
    render_html,
    render_markdown,
)

from .factories import (
    REVIEW_DIVERGENCE,
    SLO,
    TTFT_SGLANG,
    assert_self_contained,
    eval_row,
    gate_row,
    make_runs,
    review_details,
)


@pytest.fixture(scope="module")
def results(all_runs, price_book):
    single = make_runs(
        "sglang-1rep", engine="sglang", ttft=TTFT_SGLANG, reps=(1,), latency_scale=0.5
    )
    out = analyze_runs(
        [*all_runs, *single],
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )
    h = {r.name: r.config_hash for r in out}
    evals = [
        eval_row(h["vllm-bf16"], "gsm8k", 0.80),
        eval_row(h["sglang-bf16"], "gsm8k", 0.79),
        eval_row(h["vllm-awq"], "gsm8k", 0.70),
    ]
    gates = [
        gate_row(h["vllm-bf16"], h["sglang-bf16"], "pass"),
        gate_row(h["vllm-bf16"], h["vllm-awq"], "fail"),
    ]
    return with_quality(out, evals, gates)


@pytest.fixture(scope="module")
def report(results, price_book):
    sglang = next(r for r in results if r.name == "sglang-bf16")
    cold = {sglang.config_hash: ColdStartStat(median_s=95.4, n=3)}
    return build_leaderboard(results, cold_starts=cold, price_book=price_book)


def test_one_board_per_model_and_workload(report):
    (board,) = report.boards
    assert board.model == "Qwen/Qwen3-8B"
    assert board.workload == "chat"
    assert board.content.value == "realistic"


def test_ranking_cheapest_first_then_gate_failed_then_untrusted(report):
    rows = report.boards[0].rows
    assert report.boards[0].price_columns == [PriceColumn.SPOT, PriceColumn.AS_RUN]
    assert [(r.rank, r.result.name, r.status) for r in rows] == [
        (1, "sglang-bf16", RowStatus.RANKED),
        (2, "vllm-bf16", RowStatus.RANKED),
        (None, "vllm-awq", RowStatus.GATE_FAILED),
        (None, "sglang-1rep", RowStatus.UNTRUSTED),
    ]
    # the gated and untrusted rows are cheaper than the leader, and still not ranked
    leader_cost = rows[0].result.cost.output_per_mtok.value
    assert rows[2].result.cost.output_per_mtok.value < leader_cost
    assert rows[3].result.cost.output_per_mtok.value < leader_cost


def test_recommendations_are_generated_from_numbers(report):
    recs = {r.result.name: r.recommendation for r in report.boards[0].rows}
    assert recs["sglang-bf16"] == (
        "Cheapest at SLO; 33% cheaper than vllm-bf16; passed the quality gate vs vllm-bf16"
    )
    assert recs["vllm-bf16"] == (
        "50% more expensive than sglang-bf16 at SLO; 14% lower p95 TTFT at goodput; "
        "quality baseline"
    )
    assert recs["vllm-awq"] == (
        "Not ranked: failed the quality gate vs vllm-bf16 (worst: gsm8k -0.100); "
        "would be 25% cheaper than sglang-bf16"
    )
    assert recs["sglang-1rep"] == (
        "Not ranked: untrusted (goodput not bracketed, single repetition); rerun before "
        "relying on it; would be 26% cheaper than sglang-bf16"
    )


def test_single_ranked_and_no_cost_recommendations(price_book):
    results = analyze_runs(
        [*make_runs("only"), *make_runs("slow", latency_scale=10)],
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )
    rows = build_leaderboard(results).boards[0].rows
    assert [(r.result.name, r.status) for r in rows] == [
        ("only", RowStatus.RANKED),
        ("slow", RowStatus.NO_COST),
    ]
    assert rows[0].recommendation == "Only ranked config at SLO; quality not evaluated"
    assert rows[1].recommendation == (
        "No cost at SLO: no tested load met the SLO; test lower loads"
    )


def test_markdown_table_and_footer(report):
    md = render_markdown(report)
    assert "| " + " | ".join(md_headers(report.boards[0])) + " |" in md
    assert "$/1M out at SLO, on-demand | $/1M in at SLO, on-demand | $/1M out at SLO, spot | " in md
    assert "$/1M out at SLO, as run | Goodput out tok/s per replica" in md
    row = next(line for line in md.splitlines() if line.startswith("| 1 |"))
    cells = [c.strip() for c in row.strip("|").split(" | ")]
    assert cells[1].startswith("**sglang-bf16**<br>sglang 0.5.21 · unquantized · 1×L40S")
    assert cells[2] == "$0.8717 [0.8503, 0.8936]"  # on-demand, 200 GB volume included
    assert cells[4] == "$0.8613 [0.8402, 0.8830]"  # spot from the price book
    assert cells[5] == "$0.8717 [0.8503, 0.8936]"  # as run: an on-demand host
    assert cells[6] == "600.0 [585.3, 615.1]"  # geometric mean, log-t CI
    assert cells[8] == "6 req/s"
    assert cells[9] == "413 [393, 434] ms"
    assert cells[11] == "800.0 [780.4, 820.2] at 8 req/s"
    assert cells[12] == "-0.010 (gsm8k) · pass"
    assert cells[13] == "95 s (median of 3)"
    assert "- sglang-1rep: load 2 req/s: 1 completed repetition; no confidence interval" in md
    for needle in (
        "## Methodology and provenance",
        "- **SLO:** TTFT p95 ≤ 600 ms, TPOT p95 ≤ 60 ms, error rate ≤ 1.00%",
        "- **Load generation:** open loop",
        "- **Content:** realistic",
        "- **Dataset:** ShareGPT_V3 (license: apache-2.0)",
        "- **Repetitions:** 1–3 per load point",
        "two-sided 95% Student-t interval",
        "the value shown is the geometric mean",
        "  - latency percentiles and means (ms): geometric mean, Student-t interval on the "
        "log scale (log_t)",
        "  - error rate, SLO attainment, cache fractions and hit rates: arithmetic mean of "
        "per-run proportions, Student-t interval clipped to [0, 1] (t_clipped)",
        "- **Cost allocation:** all_output",
        "- **Price book last checked:** 2026-10-05",
        "aws/us-east-1 g6e.xlarge + 200 GB block storage, from the price book: on-demand "
        "$1.8829/h; spot $1.8605/h; committed 1y n/a",
        "the spot price is an indicative average and moves hourly",
        "As run: $1.7219/h, spot host: spot $1.700000/h observed at launch in us-east-1a "
        "(2026-10-01T00:00:00+00:00) + 200 GB block storage",
        "As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB block storage",
        "commit `0123456789abcdef0123456789abcdef01234567`",
        "digest `sha256:8a8a",
        "Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218",
        "CUDA 13.0, driver 580.65",
    ):
        assert needle in md, needle
    for row in report.boards[0].rows:
        assert f"`bench reproduce {row.result.reproduce_run_id()}`" in md


def test_csv_has_integer_micros_and_formatted_usd(report):
    text = render_csv(report)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0]) == CSV_COLUMNS
    assert [r["config"] for r in rows] == ["sglang-bf16", "vllm-bf16", "vllm-awq", "sglang-1rep"]
    first = rows[0]
    micros = int(first["on_demand_output_per_mtok_micros"])
    assert first["on_demand_output_per_mtok_usd"] == f"${micros / 1e6:.6f}"
    assert (
        int(first["on_demand_output_per_mtok_lo_micros"])
        < micros
        < int(first["on_demand_output_per_mtok_hi_micros"])
    )
    assert first["on_demand_input_per_mtok_micros"] == ""
    assert first["on_demand_input_per_mtok_na_reason"] == "all cost allocated to output"
    assert first["on_demand_output_per_mtok_na_reason"] == ""
    assert first["goodput_output_tok_s_ci_method"] == "log_t"
    assert first["on_demand_hourly_micros"] == "1882918"
    assert first["on_demand_hourly_usd"] == "$1.882918"
    assert first["spot_hourly_micros"] == "1860518"
    assert first["committed_1y_hourly_micros"] == ""
    assert first["committed_1y_output_per_mtok_micros"] == ""
    assert first["as_run_hourly_micros"] == "1882918"
    assert rows[2]["as_run_hourly_micros"] == "1721918"  # vllm-awq ran on spot
    assert int(rows[2]["as_run_output_per_mtok_micros"]) < int(
        rows[2]["on_demand_output_per_mtok_micros"]
    )
    assert first["storage_gb"] == "200"
    assert first["as_run_price_basis"].startswith("$1.8829/h, on_demand host")
    assert (
        first["rank"] == "1"
        and rows[2]["rank"] == ""
        and rows[2]["status"] == "quality gate failed"
    )
    assert first["quality_gate"] == "pass" and rows[2]["quality_gate"] == "fail"
    assert first["cold_start_median_s"] == "95.4"
    assert first["goodput_load"] == "6.0" and first["first_failing_load"] == "8.0"
    assert first["trusted"] == "True" and rows[3]["trusted"] == "False"
    assert first["reproduce_command"].startswith("bench reproduce ")
    assert first["slo"] == "TTFT p95 ≤ 600 ms, TPOT p95 ≤ 60 ms, error rate ≤ 1.00%"
    assert first["price_last_checked"] == "2026-10-04"
    assert first["dataset"] == "ShareGPT_V3 (license: apache-2.0)"
    assert len(first["provenance_digests"].split()) == 12


ALLOWED_URL_PREFIXES = (
    "https://b0.p.awsstatic.com/pricing/",
    "https://aws.amazon.com/ebs/pricing/",
    "https://instances.vantage.sh/",
    "https://huggingface.co/datasets/",
)


def test_html_is_self_contained_with_dark_mode(report):
    html = render_html(report)
    assert_self_contained(html, ALLOWED_URL_PREFIXES)
    assert "@media (prefers-color-scheme: dark)" in html
    assert "<title>Loom leaderboard: cost at SLO</title>" in html
    assert '<span class="badge gate_failed">quality gate failed</span>' in html
    assert "Cheapest at SLO; 33% cheaper than vllm-bf16" in html
    assert "Methodology and provenance" in html
    for row in report.boards[0].rows:
        assert f"<code>bench reproduce {row.result.reproduce_run_id()}</code>" in html


def test_review_is_ranked_and_flagged(results, price_book):
    h = {r.name: r.config_hash for r in results}
    gates = [
        gate_row(h["vllm-bf16"], h["sglang-bf16"], "review", review_details()),
        gate_row(h["vllm-bf16"], h["vllm-awq"], "fail"),
    ]
    evals = [eval_row(h["vllm-bf16"], "gsm8k", 0.80), eval_row(h["sglang-bf16"], "gsm8k", 0.80)]
    report = build_leaderboard(with_quality(results, evals, gates), price_book=price_book)
    rows = {r.result.name: r for r in report.boards[0].rows}
    sglang = rows["sglang-bf16"]
    assert (sglang.rank, sglang.status) == (1, RowStatus.RANKED)
    divergence = REVIEW_DIVERGENCE
    assert sglang.recommendation.endswith(
        f"quality needs review vs vllm-bf16: tasks pass but logprob divergence ({divergence})"
    )
    assert "+0.000 (gsm8k) · needs review" in render_markdown(report)
    html = render_html(report)
    assert '<span class="badge review">needs review</span>' in html
    csv_rows = {r["config"]: r for r in csv.DictReader(io.StringIO(render_csv(report)))}
    assert csv_rows["sglang-bf16"]["quality_gate"] == "review"
    assert csv_rows["sglang-bf16"]["quality_divergence"] == divergence
    assert csv_rows["vllm-awq"]["quality_divergence"] == ""


def test_html_escapes_untrusted_text(results, price_book):
    hostile = results[0].model_copy(update={"cell_key": "<script>x</script>"})
    html = render_html(build_leaderboard([hostile], price_book=price_book))
    assert "<script>" not in html
    assert "&lt;script&gt;x&lt;/script&gt;" in html


def test_unallocated_price_is_na_not_zero(report):
    md = render_markdown(report)
    row = next(line for line in md.splitlines() if line.startswith("| 1 |"))
    cells = [c.strip() for c in row.strip("|").split(" | ")]
    assert cells[3] == "n/a (all cost allocated to output)"
    assert "$0.0000" not in md
    html = render_html(report)
    assert 'n/a<span class="ci">all cost allocated to output</span>' in html
    assert "$0.0000" not in html


def test_all_input_ranks_by_input_cost(all_runs, price_book):
    results = analyze_runs(
        all_runs,
        slo=SLO,
        allocation=CostAllocation.all_input(),
        price_resolver=default_price_resolver(price_book),
    )
    rows = build_leaderboard(results).boards[0].rows
    ranked = [r for r in rows if r.status is RowStatus.RANKED]
    assert [r.result.name for r in ranked] == ["vllm-awq", "sglang-bf16", "vllm-bf16"]
    assert all(r.result.cost.output_per_mtok.na_reason for r in rows)
    costs = [r.result.cost.input_per_mtok.value for r in ranked]
    assert costs == sorted(costs)


def test_unbracketed_goodput_is_marked_and_explained_once(report):
    md = render_markdown(report)
    rows = {
        cells[1].split("**")[1]: cells
        for line in md.splitlines()
        if line.startswith("| ") and "**" in line
        for cells in [[c.strip() for c in line.strip("|").split(" | ")]]
    }
    assert rows["sglang-bf16"][8] == "6 req/s"  # bracketed: a failing load above it
    assert rows["sglang-1rep"][8] == "8+ req/s"  # every tested load met the SLO
    assert md.count(UNBRACKETED_NOTE) == 1
    assert render_html(report).count(UNBRACKETED_NOTE) == 1


def test_cold_start_keeps_two_significant_figures():
    assert cold_text(ColdStartStat(median_s=0.23, n=1)) == "0.23 s (median of 1)"
    assert cold_text(ColdStartStat(median_s=1.3282, n=2)) == "1.3 s (median of 2)"
    assert cold_text(ColdStartStat(median_s=95.4, n=3)) == "95 s (median of 3)"
    assert cold_text(ColdStartStat(median_s=123.4, n=1)) == "123 s (median of 1)"


def test_provenance_is_per_workload_not_per_config(price_book):
    runs = [*make_runs("vllm-bf16"), *make_runs("vllm-bf16", workload="code")]
    results = analyze_runs(
        runs,
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )
    assert len({r.config_hash for r in results}) == 1 and len(results) == 2
    report = build_leaderboard(results, price_book=price_book)
    rows = list(csv.DictReader(io.StringIO(render_csv(report))))
    by_workload = {r["workload"]: r["reproduce_command"] for r in rows}
    expected = {r.workload: f"bench reproduce {r.reproduce_run_id()}" for r in results}
    assert by_workload == expected and len(set(by_workload.values())) == 2
    md = render_markdown(report)
    assert "### vllm-bf16 · chat, open loop (`" in md
    assert "### vllm-bf16 · code, open loop (`" in md
