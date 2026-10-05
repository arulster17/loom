"""The summary after `bench run` uses the report's analysis, so trust never disagrees."""

from types import SimpleNamespace

from rich.console import Console
from site_helpers import SLO, make_runs

from loom_bench import cli
from loom_bench.cost import CostAllocation
from loom_bench.experiment import load_experiment
from loom_bench.prices import load_prices
from loom_bench.registry import REPO_ROOT
from loom_bench.report import analyze_runs, default_price_resolver
from loom_bench.site import export_snapshot, load_snapshot
from loom_bench.store.db import session_scope


def _render(monkeypatch, results) -> str:
    recorded = Console(record=True, width=240)
    monkeypatch.setattr(cli, "console", recorded)
    cli.render_goodput(results)
    return recorded.export_text()


def test_run_summary_is_the_report_analysis(populated, tmp_path):
    exp = load_experiment(REPO_ROOT / "bench/experiments/mock-smoke.yaml").model_copy(
        update={"slo": SLO}
    )
    ctx = SimpleNamespace(db_url=populated.url, prices=load_prices())
    outcome = SimpleNamespace(experiment_id=populated.experiment_id, run_ids=["any"])
    results = cli._run_results(exp, ctx, outcome)  # type: ignore[arg-type]

    # `bench report` analyses through the same function; the site snapshot through analyze_runs
    with session_scope(populated.url) as s:
        report_results = cli._analyze_experiments(
            s, [populated.experiment_id], SLO, CostAllocation.all_output(), load_prices()
        )
        manifest = export_snapshot(s, tmp_path, [populated.experiment_id])
    snapshot = load_snapshot(tmp_path).results

    def verdicts(rs):
        return sorted((r.name, r.trusted, [w.message for w in r.warnings]) for r in rs)

    assert verdicts(results) == verdicts(report_results) == verdicts(snapshot)
    assert manifest.run_count == sum(len(r.run_ids) for r in results)
    assert {r.quality.gate for r in results if r.quality} == {"baseline", "review", "fail"}


def test_summary_shows_trust_and_main_warning(monkeypatch):
    runs = [
        *make_runs("steady"),
        *make_runs("one-shot", reps=(1,), loads=(2.0, 4.0)),
    ]
    results = analyze_runs(
        runs,
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(load_prices()),
    )
    steady, one_shot = sorted(results, key=lambda r: r.name != "steady")
    assert steady.trusted and steady.main_warning is None
    assert not one_shot.trusted
    assert one_shot.main_warning.kind == "single_repetition"

    text = _render(monkeypatch, results)
    lines = {line.split("│")[1].strip(): line for line in text.splitlines() if "│" in line}
    assert "yes" in lines["steady"]
    assert "no" in lines["one-shot"]
    assert "single repetition" in lines["one-shot"]
    # goodput not bracketed: "+" on the load, explained once in the caption
    assert "4+ req/s" in lines["one-shot"]
    assert text.count("+ after a goodput load") == 1
