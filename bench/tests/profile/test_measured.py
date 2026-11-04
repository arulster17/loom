"""Stored runs against the floor: reconstruction, achieved share, fits, the report."""

import hashlib
import json
import sqlite3

import pytest
from typer.testing import CliRunner

from loom_bench.cli import app
from loom_bench.profile.measured import (
    default_roots,
    fit_steps,
    load_runs,
    observe_runs,
    summarize_points,
)
from loom_bench.profile.report import build_report, render_markdown, render_svg
from loom_bench.profile.roofline import ServingSetup, decode_step

from .conftest import BY_ID, steady_requests, summary


def _floor(model_id: str, batch: int, kv: float, **patch: object) -> float:
    spec = BY_ID[model_id].model_copy(update=patch)
    return decode_step(ServingSetup.from_spec(spec), batch, kv).total_ms


def test_reconstruction_recovers_batch_context_and_floor(store) -> None:
    store.add_run(
        cell="bf16", model_id="qwen3-8b", load=1.0, records=steady_requests(8), window_s=3.0
    )
    obs, skipped = observe_runs(load_runs(store.url, [store.experiment_id.hex[:8]]))
    assert skipped == []
    (o,) = obs
    assert o.batch_mean == pytest.approx(8)
    # context grows from 1000 to 1100 over decode; the window sees most of it
    assert 1000 < o.context_mean < 1100
    assert o.floor_ms == pytest.approx(_floor("qwen3-8b", 8, 8 * o.context_mean), rel=1e-3)
    assert o.itl_p50_ms == pytest.approx(30.0)
    assert o.achieved_step == pytest.approx(o.floor_ms / 30.0)
    assert o.prefill_share > 0  # 8 prompts of 1000 tokens prefilled inside the window


def test_db_is_opened_read_only(store) -> None:
    store.add_run(
        cell="bf16", model_id="qwen3-8b", load=1.0, records=steady_requests(2), window_s=3.0
    )
    before = hashlib.sha256(store.path.read_bytes()).hexdigest()
    observe_runs(load_runs(store.url, [str(store.experiment_id)]))
    assert hashlib.sha256(store.path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="SQLite"):
        load_runs("postgresql://x/y", ["abcd"])
    with pytest.raises(ValueError, match="matches 0"):
        load_runs(store.url, ["ffffffff"])


def test_relative_request_paths_resolve_next_to_the_db(store) -> None:
    store.add_run(
        cell="bf16",
        model_id="qwen3-8b",
        load=1.0,
        records=steady_requests(2),
        window_s=3.0,
        relative_uri=True,
    )
    runs = load_runs(store.url, [str(store.experiment_id)])
    assert observe_runs(runs)[1]  # not found from the cwd
    obs, skipped = observe_runs(runs, roots=default_roots(store.url))
    assert len(obs) == 1 and skipped == []


def test_fit_recovers_fixed_and_per_sequence_cost(store) -> None:
    """Step time = 12 ms + 0.5 ms per sequence: the fit returns exactly that."""
    for b in (4, 8, 16):
        itl = (12 + 0.5 * b) / 1000
        store.add_run(
            cell="c",
            model_id="qwen3-8b",
            load=float(b),
            records=steady_requests(b, itl_s=itl),
            window_s=99 * itl,
        )
    obs, _ = observe_runs(load_runs(store.url, [str(store.experiment_id)]))
    (fit,) = fit_steps(summarize_points(obs))
    assert fit.fixed_ms == pytest.approx(12.0, abs=0.05)
    assert fit.per_seq_ms == pytest.approx(0.5, abs=0.01)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.weights_floor_ms == pytest.approx(17.52, abs=0.01)
    assert fit.batch_at(22.0) == pytest.approx(20.0, abs=0.2)


def test_ratio_explanation_and_report(store) -> None:
    """bf16 and fp8-kv8 cells with goodput.json: the report explains the ratio and
    names what failed past each goodput."""
    for cell, model, patch, fixed, per in (
        ("bf16", "qwen3-8b", {}, 22.0, 0.33),
        ("fp8-kv8", "qwen3-8b-fp8", {"kv_cache_dtype": "fp8"}, 14.5, 0.19),
    ):
        for b in (8, 16, 32, 48):
            itl = (fixed + per * b) / 1000
            failing = b == 48
            store.add_run(
                cell=cell,
                model_id=model,
                load=float(b),
                records=steady_requests(b, itl_s=itl),
                window_s=99 * itl,
                summ=summary(
                    99 * itl,
                    tpot_p95=60.0 if failing else 40.0,
                    kv_usage_max=0.5,
                    kv_usage_mean=0.4,
                ),
                **patch,
            )
    exp_dir = store.root / str(store.experiment_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "goodput.json").write_text(
        json.dumps(
            [
                {"cell": "bf16", "workload": "fixed-1k-1k", "max_load": 16.0},
                {"cell": "fp8-kv8", "workload": "fixed-1k-1k", "max_load": 32.0},
            ]
        )
    )
    rep = build_report(
        store.url, [str(store.experiment_id)], [], compare=[("fixed-1k-1k", "bf16", "fp8-kv8")]
    )
    (x,) = rep.ratios
    assert x.measured_ratio == pytest.approx(2.0)
    assert x.fixed_a_ms == pytest.approx(22.0, abs=0.1)
    assert x.per_seq_b_ms == pytest.approx(0.19, abs=0.01)
    assert x.decode_ratio > 2.0
    assert x.fail_a is not None and x.fail_a["load"] == 32.0
    assert any("TPOT p95" in f for f in x.fail_b["failing"])
    assert x.kv_capacity_a is not None
    md = render_markdown(rep)
    assert "Why fp8-kv8 serves 2.00x bf16 on fixed-1k-1k" in md
    assert "**G**" in md
    svg = render_svg(rep.at_goodput())
    assert svg.startswith("<svg") and "bf16 · fixed-1k-1k @ 16" in svg  # cell · workload @ load


def _cells(store, load_mode: str = "open_loop", bf16_only: bool = False) -> None:
    for cell, model, patch, fixed, per in (
        ("bf16", "qwen3-8b", {}, 22.0, 0.33),
        ("fp8-kv8", "qwen3-8b-fp8", {"kv_cache_dtype": "fp8"}, 14.5, 0.19),
    ):
        if bf16_only and cell != "bf16":
            continue
        for b in (8, 16, 32):
            itl = (fixed + per * b) / 1000
            store.add_run(
                cell=cell,
                model_id=model,
                load=float(b),
                records=steady_requests(b, itl_s=itl),
                window_s=99 * itl,
                load_mode=load_mode,
                **patch,
            )


def _goodput(store, rows: list[tuple[str, float]]) -> None:
    exp_dir = store.root / str(store.experiment_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "goodput.json").write_text(
        json.dumps([{"cell": c, "workload": "fixed-1k-1k", "max_load": g} for c, g in rows])
    )


def test_ratio_pairs_cells_within_one_experiment(store) -> None:
    """A second experiment with the same cell names (a profiling run at fixed points) must
    not lend its goodput or fit to the first one's ratio, and a closed-loop pair is not
    explained (its goodput is a concurrency)."""
    _cells(store)
    _goodput(store, [("bf16", 8.0), ("fp8-kv8", 16.0)])
    sweep = str(store.experiment_id)
    store.new_experiment()
    _cells(store, bf16_only=True)
    _goodput(store, [("bf16", 32.0)])
    other = str(store.experiment_id)
    store.new_experiment()
    _cells(store, load_mode="closed_loop")
    _goodput(store, [("bf16", 8.0), ("fp8-kv8", 32.0)])
    closed = str(store.experiment_id)
    rep = build_report(
        store.url, [other, sweep, closed], [], compare=[("fixed-1k-1k", "bf16", "fp8-kv8")]
    )
    (x,) = rep.ratios
    assert x.measured_ratio == pytest.approx(2.0)  # the sweep's 16 / 8, nothing mixed in
    assert any("closed loop" in s for s in rep.skipped)


def test_cli_roofline_writes_the_report(store, tmp_path) -> None:
    store.add_run(
        cell="bf16", model_id="qwen3-8b", load=1.0, records=steady_requests(4), window_s=3.0
    )
    out = tmp_path / "rep"
    result = CliRunner().invoke(
        app, ["roofline", "-e", store.experiment_id.hex[:8], "--db", store.url, "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    md = (out / "roofline.md").read_text()
    assert "## Floors in theory" in md and "8B fp8-kv8" in md and "70B bf16 TP4 L40S" in md
    doc = json.loads((out / "roofline.json").read_text())
    assert doc["points"][0]["run_ids"] and doc["points"][0]["cell"] == "bf16"
    assert (out / "roofline.svg").exists()


def test_cli_roofline_theory_only_needs_no_db(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LOOM_DATABASE_URL", raising=False)
    result = CliRunner().invoke(app, ["roofline", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Stored runs" not in (tmp_path / "roofline.md").read_text()
    bad = CliRunner().invoke(app, ["roofline", "-e", "abcd", "--out", str(tmp_path)])
    assert bad.exit_code == 2


def test_speculative_runs_are_not_scored(store) -> None:
    store.add_run(
        cell="eagle",
        model_id="qwen3-8b",
        load=1.0,
        records=steady_requests(4),
        window_s=3.0,
        engine=BY_ID["qwen3-8b"].engine.model_copy(
            update={"args": {"speculative_config": {"method": "eagle3"}}}
        ),
    )
    obs, _ = observe_runs(load_runs(store.url, [str(store.experiment_id)]))
    (p,) = summarize_points(obs)
    assert p.speculative
    assert fit_steps([p, p, p]) == []
    assert "n/a (spec.)" in render_markdown(build_report(store.url, [str(store.experiment_id)], []))


def test_db_rows_untouched_by_the_report(store) -> None:
    store.add_run(
        cell="bf16", model_id="qwen3-8b", load=1.0, records=steady_requests(2), window_s=3.0
    )
    con = sqlite3.connect(store.path)
    n = con.execute("select count(*) from bench_runs").fetchone()[0]
    con.close()
    build_report(store.url, [str(store.experiment_id)], [])
    con = sqlite3.connect(store.path)
    assert con.execute("select count(*) from bench_runs").fetchone()[0] == n
    con.close()
