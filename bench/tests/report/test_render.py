"""End to end: store rows written and read back through the database, then every report."""

from loom_bench.cost import CostAllocation
from loom_bench.provenance import GitInfo
from loom_bench.registry import load_registry
from loom_bench.report import (
    analyze_runs,
    default_price_resolver,
    render_compare,
    render_competitiveness,
    render_leaderboard,
    write_reports,
)
from loom_bench.store.db import session_scope, upgrade
from loom_bench.store.repo import create_experiment, list_runs, record_run

from .factories import SLO


def test_reports_from_stored_runs(tmp_path, all_runs, price_book, competitors):
    url = f"sqlite:///{tmp_path / 'loom.db'}"
    upgrade(url)
    with session_scope(url) as s:
        exp = create_experiment(s, name="t", spec={}, git=GitInfo(), budget_micros=None)
        for run in all_runs:
            record_run(
                s,
                experiment_id=exp.id,
                config_hash=run.config_hash,
                provenance=run.provenance,
                status=run.status,
                summary=run.summary,
                cell_key=run.cell_key,
                workload=run.workload,
                load_mode=run.load_mode,
                load_value=run.load_value,
                repetition=run.repetition,
            )
        exp_id = exp.id
    with session_scope(url) as s:
        stored = list_runs(s, experiment_id=exp_id)
        results = analyze_runs(
            stored,
            slo=SLO,
            allocation=CostAllocation.all_output(),
            hourly_price=default_price_resolver(price_book),
        )
        run_ids = {str(r.id) for r in stored}

    assert sorted(r.name for r in results) == ["sglang-bf16", "vllm-awq", "vllm-bf16"]
    assert all(set(r.run_ids) <= run_ids for r in results)
    assert all(r.experiment_ids == [str(exp_id)] for r in results)

    leaderboard = render_leaderboard(results, price_book=price_book)
    competitiveness = render_competitiveness(results, load_registry(), competitors)
    reproduction = render_compare(results, results, label_a="original", label_b="reproduction")
    assert set(leaderboard) == {"md", "html", "csv"} == set(competitiveness)
    assert set(reproduction) == {"md", "json"}
    assert "**Verdict: within normal variance**" in reproduction["md"]
    for r in results:
        assert f"bench reproduce {r.reproduce_run_id()}" in leaderboard["md"]

    paths = write_reports(
        tmp_path / "out",
        leaderboard=leaderboard,
        competitiveness=competitiveness,
        compare=reproduction,
    )
    assert sorted(p.name for p in paths) == [
        "compare.json",
        "compare.md",
        "competitiveness.csv",
        "competitiveness.html",
        "competitiveness.md",
        "leaderboard.csv",
        "leaderboard.html",
        "leaderboard.md",
    ]
    assert (tmp_path / "out" / "leaderboard.md").read_text(encoding="utf-8") == leaderboard["md"]
    assert all(r.reproduce_run_id() in run_ids for r in results)
