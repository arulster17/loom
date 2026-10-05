"""`bench`: the Benchmark Lab command line.

Exit codes: 0 ok, 1 failed, 2 invalid input, 3 refused by the planner (over a
cap), 4 stopped before a step that would pass the cap, 5 hard budget abort,
6 reproduction outside normal variance.
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
from rich.panel import Panel
from rich.table import Table

from loom_bench.budget import load_budget
from loom_bench.experiment import ExpansionError, Experiment, load_experiment
from loom_bench.money import format_usd
from loom_bench.plan import Plan
from loom_bench.prices import load_prices
from loom_bench.registry import load_registry
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
    render_outcome(outcome)
    raise typer.Exit(outcome.exit_code)


def render_reproduce(result: ReproduceOutcome) -> None:
    for w in result.warnings:
        err.print(Panel(w, title="[bold yellow]WARNING", border_style="yellow"))
    render_outcome(result.outcome)
    if not result.comparisons:
        console.print("[yellow]no comparison: the original or the new run has no summary")
        return
    table = Table(title=f"Original {result.original_run_id} vs new {result.new_run_id}")
    for col in ("metric", "original", "new", "rel diff", "original reps range", "verdict"):
        table.add_column(col)

    def f(v: float | None) -> str:
        return "-" if v is None else f"{v:.4g}"

    for c in result.comparisons:
        table.add_row(
            c.metric,
            f(c.original),
            f(c.new),
            "-" if c.rel_diff is None else f"{c.rel_diff:.1%}",
            "-" if c.sibling_lo is None else f"{f(c.sibling_lo)} .. {f(c.sibling_hi)}",
            "[green]within" if c.within else "[red]OUTSIDE",
        )
    console.print(table)


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


@app.command("reap")
def reap_cmd(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="List, do not terminate.")] = False,
    db: DbOpt = None,
) -> None:
    """Terminate recorded resources whose TTL passed (for when a runner died)."""
    from loom_bench.store.db import upgrade

    upgrade(db)
    reaped = asyncio.run(reap(db, dry_run=dry_run))
    table = Table(title="Expired resources")
    for col in ("provider", "resource", "experiment", "ttl", "action"):
        table.add_column(col)
    for r in reaped:
        table.add_row(
            r.provider, r.resource_id, str(r.experiment_id), r.ttl_at.isoformat(), r.action
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
