"""The plain-English summary, the headline pick, workload order and the full leaderboard
with its summary and competitiveness section."""

import csv
import io

import pytest

from loom_bench.cost import CostAllocation
from loom_bench.prices import Competitors
from loom_bench.registry import load_registry
from loom_bench.report import render_leaderboard
from loom_bench.report.analyze import analyze_runs, default_price_resolver, with_quality
from loom_bench.report.format import usd_ci
from loom_bench.report.leaderboard import (
    RowStatus,
    build_leaderboard,
    headline_row,
    quality_verified,
    workload_order,
)
from loom_bench.report.summary import (
    build_summary,
    env_note,
    quality_sentence,
    summary_html,
    summary_markdown,
)
from loom_bench.slo import Slo

from .factories import SLO, assert_self_contained, eval_row, gate_row, make_runs

TTFT_FAST_SGLANG = {2.0: 40.0, 4.0: 50.0, 6.0: 60.0, 8.0: 800.0}
TTFT_FAST_VLLM = {2.0: 40.0, 4.0: 50.0, 6.0: 700.0, 8.0: 900.0}
LOOSE = Slo(ttft_ms={"p95": 100_000}, tpot_ms={"p95": 1_000}, max_error_rate=0.01)

INCONCLUSIVE = {
    "decision": "inconclusive",
    "tasks": [
        {"task": "gsm8k", "verdict": "pass", "reason": "delta +0.03 pts: non-inferior"},
        {
            "task": "ifeval",
            "verdict": "inconclusive",
            "reason": "delta -0.80 pts [-2.03 pts, +0.37 pts], n=541: CI crosses -2.00 pts",
        },
    ],
    "divergence": {"verdict": "pass", "reason": "KL 0.0006 nats"},
    "sanity": {"verdict": "pass", "reason": "no empty outputs"},
}


def _analyze(runs, price_book, slo=SLO):
    return analyze_runs(
        runs,
        slo=slo,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )


@pytest.fixture(scope="module")
def results(price_book):
    out = _analyze(
        [
            *make_runs("vllm-bf16", ttft=TTFT_FAST_VLLM),
            *make_runs("sglang-bf16", engine="sglang", ttft=TTFT_FAST_SGLANG),
        ],
        price_book,
    )
    h = {r.name: r.config_hash for r in out}
    evals = [eval_row(h["vllm-bf16"], "gsm8k", 0.90), eval_row(h["sglang-bf16"], "gsm8k", 0.90)]
    gates = [gate_row(h["vllm-bf16"], h["sglang-bf16"], "inconclusive", INCONCLUSIVE)]
    return with_quality(out, evals, gates)


@pytest.fixture(scope="module")
def summary(results, competitors):
    report = build_leaderboard(results)
    return build_summary(report, competitors=competitors, registry=load_registry())


def test_headline_prefers_trusted_then_verified_quality_over_rank(results):
    rows = build_leaderboard(results).boards[0].rows
    assert [(r.result.name, r.rank) for r in rows] == [("sglang-bf16", 1), ("vllm-bf16", 2)]
    head = headline_row(rows)
    assert head is not None and head.result.name == "vllm-bf16"
    assert quality_verified(head.result) and not quality_verified(rows[0].result)
    # ranking itself is untouched
    assert rows[0].status is RowStatus.RANKED


def test_headline_never_quotes_a_gate_failure_and_prefers_trusted(results, price_book):
    h = {r.name: r.config_hash for r in results}
    failed = with_quality(results, [], [gate_row(h["vllm-bf16"], h["sglang-bf16"], "fail")])
    rows = build_leaderboard(failed).boards[0].rows
    assert headline_row(rows).result.name == "vllm-bf16"
    single = _analyze(
        make_runs("one-rep", engine="sglang", ttft=TTFT_FAST_SGLANG, reps=(1,)), price_book
    )
    rows = build_leaderboard([*results, *single]).boards[0].rows
    assert headline_row(rows).result.name != "one-rep"  # untrusted only as a last resort
    rows = build_leaderboard(single).boards[0].rows
    assert headline_row(rows).result.name == "one-rep"


def test_workloads_lead_with_the_representative_shape(price_book):
    assert workload_order("chat") < workload_order("fixed-1k-1k") < workload_order("a-first")
    assert workload_order("chat-sharegpt") == (0, "chat-sharegpt")
    runs = [
        run
        for w in ("code-completion", "shared-prefix", "fixed-1k-1k")
        for run in make_runs("vllm-bf16", ttft=TTFT_FAST_VLLM, workload=w)
    ]
    boards = build_leaderboard(_analyze(runs, price_book)).boards
    assert [b.workload for b in boards] == ["fixed-1k-1k", "code-completion", "shared-prefix"]


def test_summary_states_cost_split_quality_and_limit(summary, results):
    (model,) = summary.models
    assert model.model == "Qwen/Qwen3-8B"
    assert model.hardware == ["1×L40S (aws g6e.xlarge, $1.8829/h on-demand incl. 200 GB storage)"]
    (chat,) = model.workloads
    assert chat.shape == "200 input / 100 output tokens per request"
    quoted = [line for line in chat.lines if line.quoted]
    assert [line.config for line in quoted] == ["vllm-bf16"]
    vllm = next(r for r in results if r.name == "vllm-bf16")
    split = vllm.split_cost
    first = chat.points[0]
    assert first.startswith(
        f"vllm-bf16: {usd_ci(split.input_per_mtok)} per 1M input tokens, "
        f"{usd_ci(split.output_per_mtok)} per 1M output tokens, "
        f"{usd_ci(split.total_per_mtok)} per 1M tokens blended at this mix, holding the SLO "
        "up to 4 req/s (fails at 6)."
    )
    assert "sglang-bf16 ranks first but its quality is not verified" in first
    limit = next(p for p in chat.points if p.startswith("What limits vllm-bf16"))
    assert limit.startswith("What limits vllm-bf16: at 6 req/s, TTFT p95 is ")
    assert "against the 600 ms target" in limit
    market = next(p for p in chat.points if p.startswith("Market"))
    assert "OpenRouter $0.2297 (aggregator, our cost " in market
    assert "no listed price is eligible for the flags" in market
    assert model.quality == [  # in leaderboard order
        "sglang-bf16 (unquantized): quality gate vs vllm-bf16 is inconclusive: gsm8k, "
        "divergence, sanity pass; ifeval inconclusive (delta -0.80 pts [-2.03 pts, "
        "+0.37 pts], n=541: CI crosses -2.00 pts). An inconclusive gate blocks the config: "
        "its quality is not shown to match vllm-bf16.",
        "vllm-bf16 (unquantized): the reference the quality gate compares against; gsm8k 0.900.",
    ]


def test_untrusted_figure_is_quoted_with_its_reason(price_book):
    single = _analyze(
        make_runs("one-rep", engine="sglang", ttft=TTFT_FAST_SGLANG, reps=(1,)), price_book
    )
    s = build_summary(build_leaderboard(single))
    points = s.models[0].workloads[0].points
    assert "No config on this board is trusted, so this figure is indicative only." in points[0]
    caveat = next(p for p in points if p.startswith("Caveat"))
    assert caveat.startswith("Caveat: one-rep is untrusted, so the leaderboard does not rank it")
    assert "load 2 req/s: 1 completed repetition; no confidence interval" in caveat
    assert s.models[0].workloads[0].lines[0].standing == "untrusted, not ranked"


def test_no_cost_at_slo_states_the_reason_and_the_labelled_alternative(price_book):
    slow = make_runs("slow", ttft=TTFT_FAST_SGLANG, latency_scale=10)
    report = build_leaderboard(_analyze(slow, price_book))
    alt = build_leaderboard(_analyze(slow, price_book, LOOSE))
    points = build_summary(report, alt=alt).models[0].workloads[0].points
    assert points[0] == "No cost at the declared SLO: no tested load met the SLO."
    assert points[1].startswith("slow misses the SLO even at the lowest tested load, 2 req/s: ")
    assert "TPOT p95 is 300.0" in points[1] and "against the 60 ms target" in points[1]
    assert "points to the per-request latency of this config on this hardware" in points[1]
    alt_line = next(p for p in points if p.startswith("Alternative SLO"))
    assert alt_line.startswith(
        "Alternative SLO, NOT the declared one (TTFT p95 ≤ 100000 ms, TPOT p95 ≤ 1000 ms, "
        "error rate ≤ 1.00%), for reference only: slow costs "
    )
    assert "per 1M tokens blended at this mix, holding the SLO up to ≥8 req/s" in alt_line
    # every load met the loose SLO, so the alternative figure is itself untrusted
    assert any(p.startswith("Under the alternative SLO: Caveat: slow is untrusted") for p in points)
    # without the alternative, no figure at all
    plain = build_summary(report).models[0].workloads[0].points
    assert not any(p.startswith("Alternative") for p in plain)


def test_alternative_view_names_its_own_slo(price_book):
    slow = make_runs("slow", ttft=TTFT_FAST_SGLANG, latency_scale=10)
    report = build_leaderboard(_analyze(slow, price_book))
    s = build_summary(report, view_note="Alternative SLO view.")
    assert s.intro.startswith("Alternative SLO view. What one replica costs")
    assert s.models[0].workloads[0].points[0].startswith("No cost at this view's SLO")


def test_env_note_names_disabled_peer_to_peer(results):
    r = results[0]
    prov = {**r.provenance, "config": {"launch": {"env": {"NCCL_P2P_DISABLE": "1"}}}}
    tp4 = r.model_copy(update={"gpus": 4, "provenance": prov})
    assert "NCCL_P2P_DISABLE=1 (no GPU peer-to-peer on this host)" in env_note(tp4)
    assert env_note(r.model_copy(update={"provenance": prov})) is None  # one GPU: no TP
    assert env_note(tp4.model_copy(update={"provenance": r.provenance})) is None


def test_quality_sentence_without_gate_or_evals(results):
    bare = results[0].model_copy(update={"quality": None})
    assert quality_sentence(bare, {}) == f"{bare.name} (unquantized): quality not evaluated."


def test_markdown_and_html(summary):
    md = summary_markdown(summary)
    assert md.startswith("## Summary\n\nWhat one replica costs us to serve at the SLO (TTFT p95")
    assert "| Workload | Config | Standing | Goodput at SLO | $/1M input |" in md
    assert "| chat | **vllm-bf16** (quoted) | rank 2 | 4 req/s (fails at 6) |" in md
    assert "**chat** (200 input / 100 output tokens per request)" in md
    assert "**Quality**" in md
    html = summary_html(summary)
    assert '<section class="summary">' in html and "<strong>vllm-bf16</strong>" in html


def test_quantized_row_market_line_uses_its_base_models_list_prices(price_book):
    url = "https://deep.example/pricing"
    market = Competitors.model_validate(
        {
            "last_checked": "2026-10-04",
            "providers": [
                {
                    "name": "Deep",
                    "pricing_url": url,
                    "last_checked": "2026-10-04",
                    "entries": [
                        {
                            "model_id": "qwen3-8b",
                            "input_per_mtok": 50_000,
                            "output_per_mtok": 150_000,
                            "quantization": "fp8",
                            "source": url,
                        }
                    ],
                }
            ],
        }
    )
    fp8 = _analyze(
        make_runs(
            "vllm-fp8",
            quantization="fp8",
            ttft=TTFT_FAST_VLLM,
            repo="RedHatAI/Qwen3-8B-FP8-dynamic",
        ),
        price_book,
    )
    summary = build_summary(build_leaderboard(fp8), competitors=market, registry=load_registry())
    md = summary_markdown(summary)
    assert "no public list price recorded" not in md
    assert (
        "Market: public list prices of qwen3-8b (the model this config serves at another "
        "precision; providers price the model) at this mix, per 1M tokens: Deep $0.0833 "
        "(fp8, same precision as ours: like for like, our cost" in md
    )
    bf16 = _analyze(make_runs("vllm-bf16", ttft=TTFT_FAST_VLLM), price_book)
    md = summary_markdown(
        build_summary(build_leaderboard(bf16), competitors=market, registry=load_registry())
    )
    assert "Market: public list prices at this mix, per 1M tokens: Deep $0.0833 (fp8, ours " in md
    assert "unquantized: not like for like" in md


def test_full_leaderboard_has_summary_then_boards_then_competitiveness(
    results, price_book, competitors
):
    rendered = render_leaderboard(
        results, price_book=price_book, registry=load_registry(), competitors=competitors
    )
    assert set(rendered) == {"md", "html", "csv", "equal_load.csv", "competitiveness.csv"}
    md = rendered["md"]
    order = [
        md.index("## Summary"),
        md.index("## Leaderboards"),
        md.index("## Competitiveness: our cost at SLO vs public list prices"),
        md.index("## Methodology and provenance"),
    ]
    assert order == sorted(order)
    assert "### Qwen3 8B (`qwen3-8b`), ours unquantized" in md
    assert "Llama 3.3 70B" not in md  # only benchmarked models
    assert "| chat (open loop) | vllm-bf16 (leaderboard rank 2) | 200 / 100 |" in md
    rows = list(csv.DictReader(io.StringIO(rendered["competitiveness.csv"])))
    assert {r["competitor"] for r in rows} == {"Fireworks AI", "OpenRouter"}
    sources = tuple(str(e.source) for p in competitors.providers for e in p.entries)
    allowed = (
        *sources,
        "https://b0.p.awsstatic.com/pricing/",
        "https://aws.amazon.com/ebs/pricing/",
        "https://instances.vantage.sh/",
        "https://huggingface.co/datasets/",
    )
    assert_self_contained(rendered["html"], allowed)
    html = rendered["html"]
    assert html.index('<section class="summary">') < html.index("<h2>Leaderboards</h2>")
    assert html.index("<h2>Leaderboards</h2>") < html.index('<section class="competitiveness">')
    # without registry and competitors: summary, no competitiveness section
    plain = render_leaderboard(results, price_book=price_book)
    assert "## Summary" in plain["md"] and "## Competitiveness" not in plain["md"]
    assert "competitiveness.csv" not in plain


def test_split_is_computed_whatever_the_declared_allocation(results, price_book):
    runs = make_runs("vllm-bf16", ttft=TTFT_FAST_VLLM)
    (declared,) = analyze_runs(
        runs,
        slo=SLO,
        allocation=CostAllocation.prefill_time(),
        price_resolver=default_price_resolver(price_book),
    )
    (default,) = _analyze(runs, price_book)
    assert declared.cost == declared.split_cost == default.split_cost
    assert default.cost.allocation == "all_output"
    assert default.split_cost.allocation == "prefill_time"
    assert default.tokens_per_request() == pytest.approx((200, 100))


def test_gate_checks_are_read_from_the_decision(results):
    sglang = next(r for r in results if r.name == "sglang-bf16")
    checks = [(c.name, c.verdict) for c in sglang.quality.checks]
    assert checks == [
        ("gsm8k", "pass"),
        ("ifeval", "inconclusive"),
        ("divergence", "pass"),
        ("sanity", "pass"),
    ]
    vllm = next(r for r in results if r.name == "vllm-bf16")
    assert vllm.quality.checks == []  # the baseline has no decision of its own


def test_leaderboard_csv_has_the_split(results, price_book):
    rendered = render_leaderboard(results, price_book=price_book)
    rows = {r["config"]: r for r in csv.DictReader(io.StringIO(rendered["csv"]))}
    vllm = next(r for r in results if r.name == "vllm-bf16")
    row = rows["vllm-bf16"]
    assert int(row["split_input_per_mtok_micros"]) == vllm.split_cost.input_per_mtok.value
    assert int(row["split_output_per_mtok_micros"]) == vllm.split_cost.output_per_mtok.value
    assert int(row["split_blended_per_mtok_micros"]) == vllm.cost.total_per_mtok.value
    assert row["split_input_time_share_ci_method"] == "log_t"
    assert float(row["split_input_time_share"]) == pytest.approx(4 * 0.05, rel=0.05)
    assert (row["tokens_in_per_request"], row["tokens_out_per_request"]) == ("200.0", "100.0")
    # the ranking column keeps the declared allocation
    assert int(row["on_demand_output_per_mtok_micros"]) == vllm.cost.output_per_mtok.value
