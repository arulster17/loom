"""`bench report --alt-slo` writes a second, clearly labelled leaderboard judged with a
replaced SLO target; the main leaderboard keeps the experiments' declared SLO."""

from typer.testing import CliRunner

from loom_bench.cli import app
from loom_bench.providers.mock import MockProvider
from loom_bench.runner import run_experiment

from .conftest import mock_experiment


async def test_alt_slo_view_is_separate_and_labelled(ctx, tmp_path):
    ctx.provider = MockProvider(hourly_micros=1_000_000)
    # The main summary quotes the alternative SLO (labelled) when no config has a cost at
    # the declared one, by design. So the declared SLO here must be met whatever the
    # machine's speed: an error-rate-only SLO (the mock never errs, so its CI is [0, 0]).
    # A latency target is judged on wall-clock time at the 95% CI upper bound, which with
    # 2 repetitions (t = 12.7) fails the default TTFT p95 <= 1000 ms under CPU load.
    experiment = mock_experiment(name="alt-slo-unit", slo={"max_error_rate": 0.01})
    outcome = await run_experiment(experiment, ctx)
    assert outcome.status.value == "completed", outcome.reason
    out = tmp_path / "rep"
    run = CliRunner().invoke(
        app,
        ["report", "--db", ctx.db_url, "--out", str(out), "--alt-slo", "tpot_ms.p95=12345"],
    )
    assert run.exit_code == 0, run.output
    main = (out / "leaderboard.md").read_text()
    alt = (out / "leaderboard.alt-slo.md").read_text()
    assert "No cost at the declared SLO" not in main  # the precondition above
    assert "Alternative SLO" not in main and "12345 ms" not in main
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
