"""CLI wiring: `bench roofline` and `bench profile ...` (registered by loom_bench.cli)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from loom_bench.registry import ModelSpec, Registry, load_registry

console = Console()
err = Console(stderr=True)
EXIT_INVALID = 2

# The configs the stored Phase 0 runs served, rebuilt from the registry: (label, model
# id, overrides). The 4x L40S row is the 70B BF16 entry as cf4d1614 / 55102ddb ran it.
DEFAULT_SETUPS: tuple[tuple[str, str, dict[str, object]], ...] = (
    ("8B bf16", "qwen3-8b", {}),
    ("8B bf16-kv8", "qwen3-8b", {"kv_cache_dtype": "fp8"}),
    ("8B fp8", "qwen3-8b-fp8", {}),
    ("8B fp8-kv8", "qwen3-8b-fp8", {"kv_cache_dtype": "fp8"}),
    ("70B bf16 TP4 L40S", "llama-3.3-70b-instruct", {"tp": 4, "gpu": "L40S"}),
    ("70B bf16 TP2 H100", "llama-3.3-70b-instruct", {}),
    ("70B fp8 TP2 H100", "llama-3.3-70b-instruct-fp8", {}),
)
# The headline comparisons, explained when their experiments are included.
DEFAULT_COMPARE = (
    ("chat-sharegpt", "bf16", "fp8-kv8"),
    ("fixed-1k-1k", "bf16", "fp8-kv8"),
    ("fixed-1k-1k", "vllm-tp2", "vllm-tp2-fp8"),
)


def _patched(spec: ModelSpec, overrides: dict[str, object]) -> ModelSpec:
    update: dict[str, object] = {}
    if "kv_cache_dtype" in overrides:
        update["kv_cache_dtype"] = overrides["kv_cache_dtype"]
    if "tp" in overrides:
        tp = int(overrides["tp"])  # type: ignore[call-overload]
        update["parallelism"] = spec.parallelism.model_copy(update={"tp": tp})
        update["hardware"] = spec.hardware.model_copy(
            update={"gpus_per_replica": tp, "gpu": overrides.get("gpu", spec.hardware.gpu)}
        )
    return spec.model_copy(update=update)


def default_setups(registry: Registry) -> list:  # list[ServingSetup]
    from loom_bench.profile.roofline import ServingSetup

    by_id = {m.id: m for m in registry.models}
    out = []
    for label, model_id, overrides in DEFAULT_SETUPS:
        spec = _patched(by_id[model_id], overrides)
        env = {"NCCL_P2P_DISABLE": "1"} if overrides.get("gpu") == "L40S" else {}
        out.append(ServingSetup.from_spec(spec, env=env, label=label))
    return out


def roofline(
    experiment: Annotated[
        list[str] | None,
        typer.Option(
            "-e", "--experiment", help="Experiment id or prefix whose runs to compare (repeatable)."
        ),
    ] = None,
    db: Annotated[
        str | None,
        typer.Option(
            "--db", help="SQLite results DB, opened read-only (default: $LOOM_DATABASE_URL)."
        ),
    ] = None,
    compare: Annotated[
        list[str] | None,
        typer.Option(
            "--compare",
            help="WORKLOAD:CELL_A:CELL_B goodput ratio to explain (repeatable; default: the "
            "Phase 0 headline pairs that are present).",
        ),
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Report directory.")] = Path(
        "reports/roofline"
    ),
) -> None:
    """Roofline floors per decode step and prefill for the Phase 0 configs, and, with
    -e, stored runs set against them (achieved share of the floor, fixed vs per-sequence
    step cost, goodput ratios explained). Writes roofline.md/.json/.svg. Free: no GPU."""
    from loom_bench.profile.report import build_report, write_report

    url = db or os.environ.get("LOOM_DATABASE_URL")
    if experiment and not url:
        err.print("[red]pass --db sqlite:///path/to/loom.db or set LOOM_DATABASE_URL[/red]")
        raise typer.Exit(EXIT_INVALID)
    pairs = []
    for c in compare or []:
        parts = c.split(":")
        if len(parts) != 3:
            err.print(f"[red]--compare {c!r}: expected WORKLOAD:CELL_A:CELL_B[/red]")
            raise typer.Exit(EXIT_INVALID)
        pairs.append((parts[0], parts[1], parts[2]))
    try:
        rep = build_report(
            url or "",
            experiment or [],
            default_setups(load_registry()),
            compare=pairs or (DEFAULT_COMPARE if experiment else ()),
        )
    except ValueError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(EXIT_INVALID) from None
    # Pairs not in the chosen experiments are expected when defaults are used.
    if not pairs:
        rep.skipped = [s for s in rep.skipped if not s.startswith("ratio ")]
    for p in write_report(rep, out):
        console.print(f"wrote {p}")


profile_app = typer.Typer(
    help="Engine profiles: trace analysis (docs/perf/roofline-and-profiling.md).",
    no_args_is_help=True,
)


@profile_app.command("analyze")
def analyze(
    trace: Annotated[
        Path,
        typer.Argument(help="A torch profiler trace (.json or .json.gz) or a .tar.gz of them."),
    ],
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the breakdown JSON here.")
    ] = None,
) -> None:
    """Break a trace's GPU time per decode step into GEMM, attention, all-reduce,
    quant/dequant, sampling, other kernels and idle gaps."""
    from loom_bench.profile.trace import analyze_path, render_breakdown

    try:
        result = analyze_path(trace)
    except (ValueError, OSError) as e:
        err.print(f"[red]cannot analyze {trace}: {e}[/red]")
        raise typer.Exit(EXIT_INVALID) from None
    console.print(render_breakdown(result))
    if out is not None:
        out.write_text(json.dumps(result.as_dict(), indent=2))
        console.print(f"wrote {out}")


def register(app: typer.Typer) -> None:
    app.command("roofline")(roofline)
    app.add_typer(profile_app, name="profile")
