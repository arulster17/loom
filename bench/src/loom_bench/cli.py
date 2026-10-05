"""`bench`: the Benchmark Lab command line.

Exit codes: 0 ok, 1 failed, 2 invalid input, 3 refused by the planner (over a
cap), 4 stopped before a step that would pass the cap, 5 hard budget abort,
6 reproduction outside normal variance, 7 quality gate blocked.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from loom_bench.budget import load_budget
from loom_bench.experiment import ExpansionError, Experiment, load_experiment
from loom_bench.money import format_usd
from loom_bench.plan import Plan
from loom_bench.prices import load_prices
from loom_bench.registry import load_registry
from loom_bench.report.analyze import ColdStartStat, ConfigResult
from loom_bench.report.compare import render_markdown as render_compare_markdown
from loom_bench.runner import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_REFUSED,
    Outcome,
    PlanRefused,
    ReproduceOutcome,
    RunnerContext,
    plan_experiment,
    reap,
    reproduce,
    run_experiment,
)

EXIT_INVALID = 2
EXIT_MISMATCH = 6
EXIT_GATE_BLOCKED = 7

app = typer.Typer(
    name="bench",
    help="Loom Benchmark Lab: reproducible latency, throughput, cost and quality benchmarks.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
db_app = typer.Typer(help="Results database.", no_args_is_help=True)
job_app = typer.Typer(help="Load jobs (run on the GPU host by cloud providers).")
app.add_typer(db_app, name="db")
app.add_typer(job_app, name="job")
quality_app = typer.Typer(help="Quality suites and the regression gate.", no_args_is_help=True)
app.add_typer(quality_app, name="quality")
site_app = typer.Typer(help="Public results site (docs/site.md).", no_args_is_help=True)
app.add_typer(site_app, name="site")
waitlist_app = typer.Typer(help="Waitlist demand signal (docs/waitlist.md).", no_args_is_help=True)
app.add_typer(waitlist_app, name="waitlist")

console = Console()
err = Console(stderr=True)

DbOpt = Annotated[
    str | None,
    typer.Option("--db", help="Database URL (default: $LOOM_DATABASE_URL, else local Postgres)."),
]
OutOpt = Annotated[Path, typer.Option("--out", help="Directory for Parquet and provenance files.")]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=err, show_path=False)],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _load(path: Path) -> Experiment:
    try:
        return load_experiment(path)
    except (ValidationError, FileNotFoundError) as e:
        err.print(f"[red]invalid experiment {path}:[/red]\n{e}")
        raise typer.Exit(EXIT_INVALID) from None


def _context(db: str | None, out: Path) -> RunnerContext:
    from loom_bench.store.db import upgrade

    upgrade(db)
    return RunnerContext(
        db_url=db,
        out_dir=out,
        registry=load_registry(),
        prices=load_prices(),
        budget=load_budget(),
    )


def _plan(exp: Experiment, ctx: RunnerContext) -> Plan:
    try:
        _, plan = plan_experiment(exp, ctx)
    except ExpansionError as e:
        err.print(f"[red]cannot expand {exp.name}:[/red] {e}")
        raise typer.Exit(EXIT_INVALID) from None
    return plan


def _secs(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def render_plan(plan: Plan) -> None:
    table = Table(title=f"Plan: {plan.experiment} ({plan.provider})", show_lines=False)
    for col in ("host", "step", "what", "time"):
        table.add_column(col)
    for host in plan.hosts:
        for i, step in enumerate(host.steps):
            table.add_row(host.key if i == 0 else "", step.kind, step.label, _secs(step.seconds))
        table.add_row(
            "",
            "[bold]host total",
            f"{format_usd(host.hourly_micros)}/h, {host.market}, TTL {_secs(host.ttl_s)}",
            f"[bold]{_secs(host.seconds)} = {format_usd(host.cost_micros, 2)}",
            end_section=True,
        )
    console.print(table)
    caps = plan.caps
    summary = Table.grid(padding=(0, 2))
    summary.add_row("cells / max runs", f"{plan.n_cells} / {plan.n_runs_max}")
    summary.add_row("estimated time", _secs(plan.total_seconds))
    summary.add_row("estimated spend", format_usd(plan.total_micros, 2))
    summary.add_row("worst case (all hosts to TTL)", format_usd(plan.ttl_worst_micros, 2))
    summary.add_row(
        "caps",
        f"experiment {format_usd(caps.max_spend, 2)}, per-experiment "
        f"{format_usd(caps.per_experiment, 2)}, overall {format_usd(caps.overall, 2)} "
        f"({format_usd(caps.overall_spent, 2)} already spent)",
    )
    summary.add_row("effective cap", f"[bold]{format_usd(caps.effective, 2)}")
    console.print(summary)
    for note in plan.notes:
        console.print(f"[yellow]note:[/yellow] {note}")
    if plan.refusals:
        console.print(
            Panel(
                "\n".join(plan.refusals),
                title="[bold red]REFUSED",
                border_style="red",
            )
        )


def render_outcome(outcome: Outcome) -> None:
    color = {"completed": "green", "aborted": "red", "failed": "red"}[outcome.status.value]
    console.print(
        f"[bold {color}]{outcome.status.value}[/] experiment {outcome.experiment_id}: "
        f"{len(outcome.run_ids)} runs, spent {format_usd(outcome.spent_micros)}"
    )
    if outcome.reason:
        console.print(f"[{color}]reason:[/] {outcome.reason}")
    for g in outcome.gates:
        verdict = "[red]BLOCKED" if g.blocked else "[green]allowed"
        console.print(f"quality gate {g.cell} vs {g.baseline}: {g.decision} ({verdict}[/])")
    if not outcome.goodput:
        return
    table = Table(title="Goodput at SLO")
    for col in ("cell", "workload", "mode", "max load", "out tok/s", "$/1M out", "trusted"):
        table.add_column(col)
    for g in outcome.goodput:
        table.add_row(
            g.cell,
            g.workload,
            g.load_mode.value,
            "-" if g.max_load is None else f"{g.max_load:g}{'' if g.bracketed else '+'}",
            "-" if g.output_tok_s is None else f"{g.output_tok_s:.1f}",
            "-" if g.output_per_mtok_micros is None else format_usd(g.output_per_mtok_micros, 4),
            "yes" if g.trusted else "no",
        )
    console.print(table)


@app.command()
def plan(experiment: Path, db: DbOpt = None) -> None:
    """Show hosts, step times and estimated spend against the caps."""
    exp = _load(experiment)
    p = _plan(exp, _context(db, Path("results")))
    render_plan(p)
    raise typer.Exit(EXIT_OK if p.ok else EXIT_REFUSED)


@app.command()
def run(
    experiment: Path,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the plan and stop.")] = False,
    db: DbOpt = None,
    out: OutOpt = Path("results"),
) -> None:
    """Plan, check the caps, then run the experiment. Refuses if over budget."""
    exp = _load(experiment)
    ctx = _context(db, out)
    p = _plan(exp, ctx)
    render_plan(p)
    if not p.ok:
        raise typer.Exit(EXIT_REFUSED)
    if dry_run:
        raise typer.Exit(EXIT_OK)
    _setup_logging()
    try:
        outcome = asyncio.run(run_experiment(exp, ctx))
    except PlanRefused as e:
        render_plan(e.plan)
        raise typer.Exit(EXIT_REFUSED) from None
    except Exception as e:  # provider setup; the experiment is already marked failed
        err.print(f"[red]experiment {exp.name} failed:[/red] {type(e).__name__}: {e}")
        raise typer.Exit(EXIT_FAILED) from None
    render_outcome(outcome)
    raise typer.Exit(outcome.exit_code)


def render_reproduce(result: ReproduceOutcome) -> None:
    for w in result.warnings:
        err.print(Panel(w, title="[bold yellow]WARNING", border_style="yellow"))
    render_outcome(result.outcome)
    if result.comparison is None:
        console.print("[yellow]no comparison: the original or the new run did not complete")
        return
    console.print(Markdown(render_compare_markdown(result.comparison)))
    verdict = result.comparison.within_normal_variance
    console.print(
        "[bold green]reproduced within normal variance"
        if verdict
        else "[bold red]NOT reproduced: outside normal variance"
    )


@app.command("reproduce")
def reproduce_cmd(
    ref: Annotated[str, typer.Argument(help="Run id, or a run's provenance.json.")],
    db: DbOpt = None,
    out: OutOpt = Path("results"),
    spec: Annotated[
        Path | None, typer.Option(help="Experiment spec, when not next to provenance.json.")
    ] = None,
    tolerance: Annotated[float, typer.Option(help="Relative difference accepted.")] = 0.25,
) -> None:
    """Re-run one stored run from its provenance and compare with the original."""
    ctx = _context(db, out)
    _setup_logging()
    try:
        result = asyncio.run(reproduce(ref, ctx, spec_path=spec, tolerance=tolerance))
    except PlanRefused as e:
        render_plan(e.plan)
        raise typer.Exit(EXIT_REFUSED) from None
    except (LookupError, ValueError, ValidationError) as e:
        err.print(f"[red]cannot reproduce {ref}:[/red] {e}")
        raise typer.Exit(EXIT_INVALID) from None
    render_reproduce(result)
    if result.outcome.exit_code != EXIT_OK:
        raise typer.Exit(result.outcome.exit_code)
    raise typer.Exit(EXIT_OK if result.ok else EXIT_MISMATCH)


def _ec2_client() -> object | None:
    """An EC2 client for the configured AWS settings, or None when AWS is not set up."""
    try:
        import boto3

        from loom_bench.providers.aws_ec2 import load_aws_settings

        settings = load_aws_settings()
    except (ImportError, ValidationError, FileNotFoundError):
        return None
    return boto3.client("ec2", region_name=settings.region)


@app.command("reap")
def reap_cmd(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="List, do not terminate.")] = False,
    db: DbOpt = None,
) -> None:
    """Terminate resources whose TTL passed (for when a runner died).

    Covers every provider's DB-recorded resources and, when AWS settings are
    configured ($LOOM_AWS_CONFIG or LOOM_AWS_*), every Loom-tagged EC2 instance.
    """
    from loom_bench.store.db import upgrade

    upgrade(db)
    ec2 = _ec2_client()
    if ec2 is None:
        console.print("[yellow]AWS not configured: only DB-recorded mock/local resources reaped")
    reaped = asyncio.run(reap(db, dry_run=dry_run, ec2=ec2))
    table = Table(title="Expired resources")
    for col in ("provider", "resource", "experiment", "ttl", "action"):
        table.add_column(col)
    for r in reaped:
        table.add_row(
            r.provider,
            r.resource_id,
            str(r.experiment_id or "-"),
            r.ttl_at.isoformat() if r.ttl_at else "-",
            r.action,
        )
    console.print(table if reaped else "no expired resources")


@db_app.command("upgrade")
def db_upgrade(db: DbOpt = None) -> None:
    """Apply schema migrations up to head."""
    from loom_bench.store.db import database_url, upgrade

    upgrade(db)
    console.print(f"database at head: {database_url(db).split('@')[-1]}")


@app.command()
def export(
    fmt: Annotated[Literal["csv", "parquet"], typer.Argument(metavar="csv|parquet")],
    out: Annotated[Path, typer.Option("--out", help="Output file.")],
    experiment: Annotated[str | None, typer.Option(help="Experiment id (default: all).")] = None,
    db: DbOpt = None,
) -> None:
    """One row per run: summary and provenance flattened into columns."""
    from loom_bench.store import repo
    from loom_bench.store.db import session_scope
    from loom_bench.store.export import export_runs_csv, export_runs_parquet

    exp_id = uuid.UUID(experiment) if experiment else None
    with session_scope(db) as s:
        runs = repo.list_runs(s, experiment_id=exp_id)
        path = (export_runs_csv if fmt == "csv" else export_runs_parquet)(runs, out)
    console.print(f"wrote {len(runs)} runs to {path}")


@job_app.command("run")
def job_run(
    in_: Annotated[Path, typer.Option("--in", help="LoadJob JSON.")],
    out: Annotated[Path, typer.Option("--out", help="Where to write the LoadJobResult JSON.")],
) -> None:
    """Execute one LoadJob here (used on GPU hosts)."""
    from loom_bench.jobexec import execute_load_job
    from loom_bench.jobs import LoadJob

    job = LoadJob.model_validate_json(in_.read_text())
    try:
        result = asyncio.run(execute_load_job(job))
    except Exception as e:
        err.print(f"[red]job {job.run_id} failed:[/red] {type(e).__name__}: {e}")
        raise typer.Exit(EXIT_FAILED) from None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json())


@app.command("mock-server")
def mock_server(
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    config: Annotated[Path | None, typer.Option(help="YAML of MockConfig fields.")] = None,
) -> None:
    """Run the OpenAI-compatible mock backend (simulated batching GPU)."""
    import uvicorn

    from loom_bench.mock.config import MockConfig
    from loom_bench.mock.server import create_app

    cfg = MockConfig.from_yaml(config) if config else MockConfig()
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


ExperimentsOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--experiment",
        "-e",
        help="Experiment id, repeatable (default: every completed experiment but reproductions).",
    ),
]
ReportDirOpt = Annotated[Path, typer.Option("--out", help="Directory for the report files.")]


def _analyze(
    db: str | None, experiments: list[str] | None
) -> tuple[list[ConfigResult], dict[str, ColdStartStat]]:
    """ConfigResults (with quality) and cold-start stats for the selected experiments."""
    from sqlalchemy import select

    from loom_bench.report import (
        analyze_runs,
        cold_starts_by_config,
        default_price_resolver,
        with_quality,
    )
    from loom_bench.store import repo
    from loom_bench.store.db import session_scope, upgrade
    from loom_bench.store.models import (
        BenchColdStart,
        BenchEvalRun,
        BenchExperiment,
        BenchGateDecision,
    )

    upgrade(db)
    with session_scope(db) as s:
        stmt = select(BenchExperiment)
        if experiments:
            stmt = stmt.where(BenchExperiment.id.in_([uuid.UUID(e) for e in experiments]))
        else:
            stmt = stmt.where(
                BenchExperiment.status == "completed",
                BenchExperiment.name.not_like("%--reproduce"),
            )
        rows = list(s.scalars(stmt))
        if not rows:
            err.print("[red]no matching experiments")
            raise typer.Exit(EXIT_INVALID)
        specs = [Experiment.model_validate(r.spec) for r in rows]
        slo = specs[0].slo
        same = all(x.slo == slo and x.cost_allocation == specs[0].cost_allocation for x in specs)
        if slo is None or not same:
            err.print(
                "[red]selected experiments must all declare the same slo and cost_allocation; "
                "pick them with --experiment"
            )
            raise typer.Exit(EXIT_INVALID)
        ids = [r.id for r in rows]
        runs = [run for i in ids for run in repo.list_runs(s, experiment_id=i)]
        results = analyze_runs(
            runs,
            slo=slo,
            allocation=specs[0].cost_allocation.allocation(),
            hourly_price=default_price_resolver(load_prices()),
        )
        evals = s.scalars(select(BenchEvalRun).where(BenchEvalRun.experiment_id.in_(ids)))
        gates = s.scalars(select(BenchGateDecision).where(BenchGateDecision.experiment_id.in_(ids)))
        results = with_quality(results, evals, gates)
        colds = s.scalars(select(BenchColdStart).where(BenchColdStart.experiment_id.in_(ids)))
        return results, cold_starts_by_config(colds, results)


def _written(paths: list[Path]) -> None:
    for p in paths:
        console.print(f"wrote {p}")


@app.command()
def report(
    experiment: ExperimentsOpt = None,
    out: ReportDirOpt = Path("reports"),
    db: DbOpt = None,
) -> None:
    """Leaderboard ranked by $/1M output tokens at SLO (md, html, csv)."""
    from loom_bench.report import render_leaderboard, write_reports

    results, cold = _analyze(db, experiment)
    rendered = render_leaderboard(results, cold_starts=cold, price_book=load_prices())
    console.print(Markdown(rendered["md"]))
    _written(write_reports(out, leaderboard=rendered))


@app.command()
def competitiveness(
    experiment: ExperimentsOpt = None,
    out: ReportDirOpt = Path("reports"),
    db: DbOpt = None,
    include_aggregators: Annotated[bool, typer.Option(help="Include resellers.")] = False,
    include_unverified: Annotated[bool, typer.Option(help="Include unverified listings.")] = False,
) -> None:
    """Our cost at SLO and price vs competitors' list prices, with margin flags."""
    from loom_bench.prices import load_competitors
    from loom_bench.report import render_competitiveness, write_reports

    results, _ = _analyze(db, experiment)
    rendered = render_competitiveness(
        results,
        load_registry(),
        load_competitors(),
        price_book=load_prices(),
        include_aggregators=include_aggregators,
        include_unverified=include_unverified,
    )
    console.print(Markdown(rendered["md"]))
    _written(write_reports(out, competitiveness=rendered))


@app.command("compare")
def compare_cmd(
    a: Annotated[str, typer.Argument(help="Experiment id (A).")],
    b: Annotated[str, typer.Argument(help="Experiment id (B).")],
    match_by: Annotated[
        Literal["config_hash", "cell_key", "workload"],
        typer.Option(help="How sweeps are paired across A and B."),
    ] = "config_hash",
    tolerance: Annotated[float, typer.Option(help="Relative difference accepted.")] = 0.10,
    out: Annotated[Path | None, typer.Option("--out", help="Write compare.md/json here.")] = None,
    db: DbOpt = None,
) -> None:
    """Per-metric deltas between two experiments; exit 6 when outside normal variance."""
    from loom_bench.report import write_reports
    from loom_bench.report.compare import compare, render_json
    from loom_bench.store import repo
    from loom_bench.store.db import session_scope

    with session_scope(db) as s:
        runs_a = repo.list_runs(s, experiment_id=uuid.UUID(a))
        runs_b = repo.list_runs(s, experiment_id=uuid.UUID(b))
    if not runs_a or not runs_b:
        err.print("[red]both experiments need runs")
        raise typer.Exit(EXIT_INVALID)
    c = compare(runs_a, runs_b, match_by=match_by, rel_tol=tolerance, label_a=a, label_b=b)
    md = render_compare_markdown(c)
    console.print(Markdown(md))
    if out is not None:
        _written(write_reports(out, compare={"md": md, "json": render_json(c)}))
    raise typer.Exit(EXIT_OK if c.within_normal_variance else EXIT_MISMATCH)


@quality_app.command("run")
def quality_run(
    suite: Annotated[str, typer.Argument(help="Suite name (bench/evals/<name>.yaml) or path.")],
    base_url: Annotated[str, typer.Option(help="OpenAI-compatible base URL, with /v1.")],
    model: Annotated[str, typer.Option(help="Served model name.")],
    out: Annotated[Path, typer.Option("--out", help="Per-item samples JSON.")] = Path(
        "quality-samples.json"
    ),
    only: Annotated[list[str] | None, typer.Option(help="Run only these tasks.")] = None,
    allow_code_exec: Annotated[
        bool, typer.Option(help="Run code_exec tasks (executes model output in the sandbox).")
    ] = False,
) -> None:
    """Run a pinned suite against one endpoint and print per-task scores."""
    from loom_bench.experiment import load_quality_suite
    from loom_bench.quality.runner import run_suite
    from loom_bench.runner import write_samples

    s = load_quality_suite(suite)
    result = asyncio.run(
        run_suite(
            s,
            base_url,
            model,
            workdir=out.parent / f"{out.stem}-work",
            allow_code_exec=allow_code_exec,
            only=only,
        )
    )
    table = Table(title=f"Suite {result.suite} on {model}")
    for col in ("task", "n", "score", "95% CI"):
        table.add_column(col)
    for name, run in result.tasks.items():
        est = run.estimate
        ci = "-" if est.lo is None else f"{est.lo:.3f} .. {est.hi:.3f}"
        table.add_row(name, str(est.n), f"{est.mean:.3f}", ci)
    console.print(table)
    console.print(f"wrote {write_samples(out, suite, result)}")


@quality_app.command("gate")
def quality_gate(
    baseline: Annotated[str, typer.Option(help="Experiment id or config hash of the baseline.")],
    candidate: Annotated[str, typer.Option(help="Experiment id or config hash to gate.")],
    suite: Annotated[
        str | None, typer.Option(help="Suite for the policy (default: the one recorded).")
    ] = None,
    db: DbOpt = None,
) -> None:
    """Re-decide the gate from stored per-item samples; exit 7 when blocked."""
    from loom_bench.runner import gate_stored

    try:
        decision, base_hash, cand_hash = gate_stored(db, baseline, candidate, suite=suite)
    except LookupError as e:
        err.print(f"[red]{e}")
        raise typer.Exit(EXIT_INVALID) from None
    console.print(f"baseline {base_hash[:12]} -> candidate {cand_hash[:12]}")
    console.print(decision.summary())
    raise typer.Exit(EXIT_GATE_BLOCKED if decision.blocked else EXIT_OK)


@site_app.command("export")
def site_export(
    experiment: Annotated[
        list[str] | None,
        typer.Option(
            "--experiment",
            "-e",
            help="Experiment id, repeatable (default: latest completed of each name).",
        ),
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Snapshot directory.")] = None,
    slo: Annotated[
        Path | None, typer.Option(help="SLO YAML, if the runs do not record one.")
    ] = None,
    db: DbOpt = None,
) -> None:
    """Write the results snapshot the site shows (default site/data)."""
    from loom_bench.site import export_snapshot
    from loom_bench.site.config import DEFAULT_SNAPSHOT_DIR
    from loom_bench.slo import Slo
    from loom_bench.store.db import session_scope

    out = out or DEFAULT_SNAPSHOT_DIR
    target = Slo.from_yaml(slo.read_text(encoding="utf-8")) if slo else None
    with session_scope(db) as s:
        manifest = export_snapshot(s, out, experiment or "latest", slo=target)
    configs = sum(m.configs for m in manifest.models)
    console.print(
        f"wrote {out}: {len(manifest.experiment_ids)} experiments, {configs} configs, "
        f"{manifest.run_count} runs"
    )


@site_app.command("build")
def site_build(
    data: Annotated[Path | None, typer.Option(help="Snapshot directory.")] = None,
    config: Annotated[Path | None, typer.Option(help="Site config YAML.")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Build directory.")] = None,
) -> None:
    """Render the static site from a snapshot (default site/data -> site/_build)."""
    from loom_bench.site import build_site
    from loom_bench.site.config import (
        DEFAULT_BUILD_DIR,
        DEFAULT_SITE_CONFIG,
        DEFAULT_SNAPSHOT_DIR,
    )

    out = out or DEFAULT_BUILD_DIR
    pages = build_site(data or DEFAULT_SNAPSHOT_DIR, out, config or DEFAULT_SITE_CONFIG)
    console.print(f"wrote {len(pages)} pages to {out}")


@waitlist_app.command("count")
def waitlist_count_cmd(
    db: DbOpt = None,
    record: Annotated[
        bool, typer.Option(help="Also add the count to the table in docs/waitlist.md.")
    ] = True,
    docs: Annotated[Path | None, typer.Option(help="Waitlist doc to record into.")] = None,
    source: Annotated[str, typer.Option(help="Source column text.")] = "waitlist_signups table",
) -> None:
    """Count signups in `waitlist_signups` and record it as a demand signal."""
    from loom_bench.site import record_count, waitlist_count
    from loom_bench.store.db import session_scope

    with session_scope(db) as s:
        count = waitlist_count(s)
    console.print(f"{count} signups")
    if record:
        console.print(f"recorded in {record_count(docs, count, source)}")
