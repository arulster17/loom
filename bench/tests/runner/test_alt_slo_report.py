"""`bench report --alt-slo` writes a second, clearly labelled leaderboard judged with a
replaced SLO target; the main leaderboard keeps the experiments' declared SLO."""

from typer.testing import CliRunner

from loom_bench.cli import app
from loom_bench.providers.mock import MockProvider
from loom_bench.runner import run_experiment

from .conftest import mock_experiment


async def test_alt_slo_view_is_separate_and_labelled(ctx, tmp_path):
    ctx.provider = MockProvider(hourly_micros=1_000_000)
    outcome = await run_experiment(mock_experiment(name="alt-slo-unit"), ctx)
    assert outcome.status.value == "completed", outcome.reason
    out = tmp_path / "rep"
    run = CliRunner().invoke(
        app,
        ["report", "--db", ctx.db_url, "--out", str(out), "--alt-slo", "tpot_ms.p95=12345"],
    )
    assert run.exit_code == 0, run.output
    main = (out / "leaderboard.md").read_text()
    alt = (out / "leaderboard.alt-slo.md").read_text()
    assert "Alternative SLO" not in main and "12345" not in main
    assert alt.startswith("# Alternative SLO view, not the experiments' SLO: ")
    assert "TPOT p95 ≤ 12345 ms" in alt and "the experiments declare" in alt
    for fmt in ("html", "csv", "equal_load.csv"):
        assert (out / f"leaderboard.alt-slo.{fmt}").exists()


async def test_a_malformed_alt_slo_is_invalid_input(ctx, tmp_path):
    ctx.provider = MockProvider(hourly_micros=1_000_000)
    await run_experiment(mock_experiment(name="alt-slo-bad"), ctx)
    run = CliRunner().invoke(
        app, ["report", "--db", ctx.db_url, "--out", str(tmp_path), "--alt-slo", "tpot=fast"]
    )
    assert run.exit_code == 2 and "invalid --alt-slo" in run.output
