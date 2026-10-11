"""One selection rule on every page: only configs that passed every check rank; a
candidate whose quality gate is inconclusive is listed but never ranked or called best;
with nothing ranked, an untrusted reference is a flagged comparison point. Configs keep
one unique name on every page, and gate references are named by what they are."""

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from site_helpers import SPEC, TTFT_SGLANG, export, make_runs, new_db, store_runs

from loom_bench.provenance import GitInfo, PriceBasis
from loom_bench.records import Market
from loom_bench.site import SiteConfig, build_site
from loom_bench.store.db import session_scope
from loom_bench.store.repo import create_experiment, record_gate_decision, update_experiment_status

INCONCLUSIVE = {
    "decision": "inconclusive",
    "tasks": [
        {
            "task": "gsm8k",
            "verdict": "inconclusive",
            "reason": "delta -0.40 pts [-1.20 pts, +0.40 pts]: CI crosses -1.00 pts",
        }
    ],
}


def _runpod(cell_key: str, **kwargs):
    runs = make_runs(
        cell_key,
        cloud="runpod",
        instance_type="l40s-x1",
        hourly_micros=1_100_959,
        price_basis=PriceBasis(market=Market.ON_DEMAND, source="observed_api", storage_gb=80),
        hardware={"cloud": "runpod"},
        **kwargs,
    )
    for run in runs:  # RunPod's price-book region (the factory prices an AWS region)
        run.provenance = {**run.provenance, "region": "secure"}
    return runs


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> Path:
    """Qwen3-8B: the BF16 reference on AWS, an FP8 + FP8 KV candidate that is cheaper but
    inconclusive, and a second "bf16" cell on RunPod. On fixed-1k-1k the reference ran a
    single repetition (untrusted), so nothing there is ranked."""
    root = tmp_path_factory.mktemp("ranking")
    url = new_db(root / "loom.db")
    runs = [
        *make_runs("bf16"),
        *make_runs("bf16", workload="fixed-1k-1k", reps=(1,)),
        *make_runs("fp8-kv8", quantization="fp8", ttft=TTFT_SGLANG),
        *make_runs("fp8-kv8", quantization="fp8", ttft=TTFT_SGLANG, workload="fixed-1k-1k"),
        *_runpod("bf16"),
    ]
    with session_scope(url) as s:
        exp = create_experiment(
            s,
            name="qwen3-8b-ranking",
            spec=SPEC,
            git=GitInfo(sha="0123456789abcdef0123456789abcdef01234567", dirty=False),
            budget_micros=50_000_000,
            created_at=datetime(2026, 10, 1, tzinfo=UTC),
        )
        hashes = {}
        for run in runs:
            hashes.setdefault((run.cell_key, run.provenance.get("cloud")), run.config_hash)
        store_runs(s, exp.id, runs)
        record_gate_decision(
            s,
            experiment_id=exp.id,
            baseline_config_hash=hashes[("bf16", "aws")],
            candidate_config_hash=hashes[("fp8-kv8", "aws")],
            decision="inconclusive",
            details=INCONCLUSIVE,
        )
        update_experiment_status(s, exp.id, "completed")
    export(url, root / "data")
    build_site(root / "data", root / "out", SiteConfig())
    return root / "out"


def _text(site: Path, page: str) -> str:
    return (site / page).read_text(encoding="utf-8")


def _rows(html: str, table_class: str) -> list[str]:
    table = html.split(f'<table class="{table_class}">', 1)[1].split("</table>", 1)[0]
    return re.findall(r"<tr[^>]*>.*?</tr>", table, flags=re.S)


def test_inconclusive_candidate_is_listed_but_never_ranked_or_best(site):
    index = _text(site, "index.html")
    assert "fp8-kv8" not in index  # never quoted, so never on the results page
    model = _text(site, "models/qwen3-8b.html")
    chat = _rows(model, "leaderboard")
    fp8 = next(r for r in chat if "fp8-kv8" in r)
    assert '<td class="num">–</td>' in fp8  # no rank
    assert '<span class="badge quality_unverified">quality not verified</span>' in fp8
    assert '<span class="badge inconclusive">inconclusive</span>' in fp8
    assert "Not ranked: quality gate inconclusive vs BF16 (AWS g6e.xlarge)" in model
    assert "Best config at SLO: <strong>bf16 (RunPod L40S)</strong>" in model


def test_nothing_ranked_shows_the_untrusted_reference_as_a_flagged_comparison_point(site):
    index = _text(site, "index.html")
    fixed = next(r for r in _rows(index, "headline") if "fixed-1k-1k" in r)
    assert "No config ranked." in fixed and "Comparison point" in fixed
    assert "<strong>bf16 (AWS g6e.xlarge)</strong>" in fixed
    assert "untrusted, not ranked" in fixed and "single repetition" in fixed
    # its numbers are shown (this fixture's overlapping prefills leave no $ split)
    assert re.search(r"\d+ ms", fixed) and "prefills overlapped at goodput" in fixed
    model = _text(site, "models/qwen3-8b.html")
    assert "Shown as a comparison point, not as the best config" in model


def test_pricing_quotes_what_the_leaderboard_quotes_with_its_basis(site):
    index = _text(site, "index.html")
    pricing = _text(site, "pricing.html")
    rows = _rows(pricing, "pricing")
    chat = next(r for r in rows if "chat" in r)
    fixed = next(r for r in rows if "fixed-1k-1k" in r)
    assert "<strong>bf16 (RunPod L40S)</strong>" in chat and "Rank 1" in chat
    assert "<strong>bf16 (AWS g6e.xlarge)</strong>" in fixed
    assert "Comparison point only, not ranked: untrusted (single repetition)" in fixed
    assert "fp8-kv8" not in pricing
    for name in ("bf16 (RunPod L40S)", "bf16 (AWS g6e.xlarge)"):
        assert f"<strong>{name}</strong>" in index


def test_labels_are_unique_and_stable_and_references_are_named(site):
    model = _text(site, "models/qwen3-8b.html")
    # two configs share the cell key "bf16": each name says where it ran
    assert "bf16 (RunPod L40S)" in model and "bf16 (AWS g6e.xlarge)" in model
    assert re.search(r"<strong>bf16</strong>", model) is None
    # the reference is named, never a bare hash
    assert re.search(r"vs [0-9a-f]{12}\b", model) is None
    assert "vs BF16 (AWS g6e.xlarge)" in model
    # details state weights and KV cache
    assert "FP8 weights" in model and "BF16 weights" in model


def test_methodology_names_each_clouds_storage(site):
    methodology = _text(site, "methodology.html")
    assert "EBS gp3 volume on AWS, container disk on RunPod" in methodology
    assert "(EBS gp3)" not in methodology
    assert "unnamed dataset" not in methodology
    assert 'id="split"' in methodology
