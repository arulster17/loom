"""Experiment and run ids on the command line: full, or a unique prefix like git's short
hashes; and `bench compare`'s exit code, which judges only the cells both sides ran."""

import json
import uuid
from datetime import UTC, datetime

import pytest
from report.factories import make_runs
from typer.testing import CliRunner

from loom_bench.cli import EXIT_INVALID, EXIT_MISMATCH, EXIT_OK, app
from loom_bench.providers.mock import MockProvider
from loom_bench.runner import _latest_eval, run_experiment
from loom_bench.store import repo
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchExperiment

from .conftest import mock_experiment

# Two ids sharing the 8-character prefix the docs use, and an unrelated one.
TWIN_A = uuid.UUID("7a8237d0-9917-47e7-b93e-8cb0230e0059")
TWIN_B = uuid.UUID("7a8237d0-1111-4111-8111-111111111111")
OTHER = uuid.UUID("565b8d3f-b521-4e3e-91e5-ee07ee02b94d")


def invoke(*args: object):
    return CliRunner().invoke(app, [str(a) for a in args])


def text(result) -> str:
    return " ".join(result.output.split())  # rich wraps long lines


def add_experiment(db: str, exp_id: uuid.UUID, runs=(), name: str = "x") -> uuid.UUID:
    with session_scope(db) as s:
        s.add(
            BenchExperiment(
                id=exp_id,
                name=name,
                spec={"name": name},
                spec_hash="0" * 64,
                git_sha=None,
                git_dirty=False,
                status="completed",
                budget_micros=None,
                spent_micros=0,
                created_at=datetime(2026, 10, 1, tzinfo=UTC),
            )
        )
        s.flush()
        for run in runs:
            repo.record_run(
                s,
                experiment_id=exp_id,
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
    return exp_id


# --- resolution --------------------------------------------------------------------


def test_resolve_full_ids_and_unique_prefixes(db):
    for exp_id in (TWIN_A, TWIN_B, OTHER):
        add_experiment(db, exp_id)
    with session_scope(db) as s:
        assert repo.resolve_experiment_id(s, str(OTHER)) == OTHER
        assert repo.resolve_experiment_id(s, OTHER) == OTHER
        assert repo.resolve_experiment_id(s, "565b8d3f") == OTHER
        assert repo.resolve_experiment_id(s, "565B") == OTHER  # case and length 4
        assert repo.resolve_experiment_id(s, "7a8237d0-9917") == TWIN_A
        assert repo.resolve_experiment_id(s, "7a8237d09917") == TWIN_A  # dashes optional
        assert repo.resolve_experiment_id(s, OTHER.hex) == OTHER
        with pytest.raises(LookupError, match=r"'7a8237d0' is ambiguous: it matches 2"):
            repo.resolve_experiment_id(s, "7a8237d0")
        with pytest.raises(LookupError, match="no experiment 0000"):
            repo.resolve_experiment_id(s, "0000")
        with pytest.raises(LookupError, match="no experiment 00000000-0000"):
            repo.resolve_experiment_id(s, "00000000-0000-0000-0000-000000000009")
        with pytest.raises(ValueError, match="too short for an experiment id"):
            repo.resolve_experiment_id(s, "565")
        with pytest.raises(ValueError, match="not an experiment id: 'not-an-id'"):
            repo.resolve_experiment_id(s, "not-an-id")


def test_resolve_run_ids(db):
    runs = make_runs("vllm-bf16", loads=(2.0,), reps=(1,))
    add_experiment(db, OTHER, runs)
    with session_scope(db) as s:
        (run,) = repo.list_runs(s, experiment_id=OTHER)
        assert repo.resolve_run_id(s, str(run.id)[:8]) == run.id
        absent = ("1" if str(run.id)[0] == "0" else "0") * 4
        with pytest.raises(LookupError, match=f"no run {absent}"):
            repo.resolve_run_id(s, absent)
        with pytest.raises(ValueError, match="not a run id"):
            repo.resolve_run_id(s, "nope")


def test_gate_refs_take_prefixes_of_experiment_ids_and_config_hashes(db):
    add_experiment(db, TWIN_A)
    add_experiment(db, OTHER)
    hash_a, hash_b = "7a8237d0" + "1" * 56, "c5e7" + "2" * 60  # hash_a shares TWIN_A's prefix
    with session_scope(db) as s:
        for exp_id, h in ((TWIN_A, hash_a), (OTHER, hash_b)):
            repo.record_eval_run(
                s,
                experiment_id=exp_id,
                config_hash=h,
                task="gsm8k",
                task_version="1",
                n=10,
                score=0.5,
                ci_low=None,
                ci_high=None,
                provenance={},
            )
    with session_scope(db) as s:
        # Both resolve, then stop at the missing samples: the ref was understood.
        for ref in ("565b8d3f", str(OTHER), "c5e7", hash_b, "7a8237d099", "7a8237d011"):
            with pytest.raises(LookupError, match=f"eval runs for {ref} have no per-item"):
                _latest_eval(s, ref)
        with pytest.raises(LookupError, match="'7a8237' is ambiguous: it matches 2"):
            _latest_eval(s, "7a8237")  # TWIN_A and hash_a
        with pytest.raises(LookupError, match="no experiment or evaluated config hash beef"):
            _latest_eval(s, "beef")
        with pytest.raises(ValueError, match="too short"):
            _latest_eval(s, "7a")


# --- the commands --------------------------------------------------------------------


def test_every_command_reports_an_ambiguous_or_unknown_prefix(db, tmp_path):
    add_experiment(db, TWIN_A)
    add_experiment(db, TWIN_B)
    for args in (
        ("report", "-e", "7a8237d0", "--out", tmp_path),
        ("competitiveness", "-e", "7a8237d0", "--out", tmp_path),
        ("compare", "7a8237d0", "7a8237d0-9917"),
        ("export", "csv", "--out", tmp_path / "x.csv", "--experiment", "7a8237d0"),
        ("site", "export", "-e", "7a8237d0", "--out", tmp_path / "snap"),
    ):
        result = invoke(*args, "--db", db)
        assert result.exit_code == EXIT_INVALID, (args, result.output)
        assert f"'7a8237d0' is ambiguous: it matches 2 ({TWIN_B}, {TWIN_A})" in text(result), args
        assert "Traceback" not in result.output
    result = invoke("report", "-e", "beef", "--out", tmp_path, "--db", db)
    assert result.exit_code == EXIT_INVALID and "no experiment beef" in text(result)
    result = invoke("reproduce", "beef", "--db", db, "--out", tmp_path)
    assert result.exit_code == EXIT_INVALID and "no run beef" in text(result)


async def test_short_ids_work_on_real_runs(ctx, tmp_path):
    """A mock experiment, then every id-taking command by an 8-character prefix."""
    ctx.provider = MockProvider(hourly_micros=1_000_000)
    outcome = await run_experiment(
        mock_experiment(name="short-ids", slo={"max_error_rate": 0.01}), ctx
    )
    assert outcome.status.value == "completed", outcome.reason
    db, short = ctx.db_url, str(outcome.experiment_id)[:8]

    report = invoke("report", "-e", short, "--db", db, "--out", tmp_path / "rep")
    assert report.exit_code == EXIT_OK, report.output
    assert (tmp_path / "rep" / "leaderboard.md").is_file()
    comp = invoke("competitiveness", "-e", short, "--db", db, "--out", tmp_path / "comp")
    assert comp.exit_code == EXIT_OK, comp.output
    export = invoke("export", "csv", "--experiment", short, "--out", tmp_path / "r.csv", "--db", db)
    assert export.exit_code == EXIT_OK, export.output
    assert f"wrote {len(outcome.run_ids)} runs" in export.output
    same = invoke("compare", short, outcome.experiment_id, "--db", db, "--out", tmp_path / "c")
    assert same.exit_code == EXIT_OK, same.output
    md = (tmp_path / "c" / "compare.md").read_text()
    assert md.startswith(f"# Comparison: {outcome.experiment_id} vs {outcome.experiment_id}")
    site = invoke("site", "export", "-e", short, "--out", tmp_path / "snap", "--db", db)
    assert site.exit_code == EXIT_OK, site.output
    manifest = json.loads((tmp_path / "snap" / "manifest.json").read_text())
    assert manifest["experiment_ids"] == [str(outcome.experiment_id)]
    assert "these are not the experiments pinned in" in text(site)


# --- compare's verdict -------------------------------------------------------------


def test_compare_exit_judges_only_the_cells_both_ran(db):
    """Like `bench compare 565b8d3f 7a8237d0`: one shared point, the rest in one side."""
    base = add_experiment(
        db,
        OTHER,
        [*make_runs("vllm-bf16"), *make_runs("sglang-bf16", engine="sglang")],
    )
    other_cell = make_runs("vllm-fp8", quantization="fp8", experiment_id=TWIN_A)
    shared = add_experiment(
        db, TWIN_A, [*make_runs("vllm-bf16", loads=(4.0,), latency_scale=1.03), *other_cell]
    )
    within = invoke("compare", "565b8d3f", "7a8237d0", "--db", db)
    assert within.exit_code == EXIT_OK, within.output
    assert "within normal variance (1 load point matched; 5 unmatched, not judged)" in text(within)

    shifted = add_experiment(
        db, TWIN_B, [*make_runs("vllm-bf16", loads=(4.0,), latency_scale=1.8), *other_cell]
    )
    outside = invoke("compare", str(base), str(shifted), "--db", db)
    assert outside.exit_code == EXIT_MISMATCH, outside.output
    assert "outside normal variance" in text(outside)

    only_other = add_experiment(db, uuid.UUID(int=5), other_cell)
    nothing = invoke("compare", str(base), str(only_other), "--db", db)
    assert nothing.exit_code == EXIT_INVALID, nothing.output
    assert "nothing to compare" in text(nothing)
    assert "Traceback" not in nothing.output
    assert shared == TWIN_A


def test_compare_across_clouds_by_cell_key(db):
    """Like `bench compare 7a8237d0 95cde129 --match-by cell_key`: the same cells on
    RunPod and AWS. The hashes differ because hardware is part of the config; that is
    listed, not judged, so the exit code follows the metrics."""
    runpod = {"provider": "runpod", "cloud": "runpod", "instance_type": "l40s-x1"}
    aws = {"provider": "aws_ec2", "cloud": "aws", "instance_type": "g6e.xlarge"}
    aws_id = uuid.UUID("95cde129-e626-4eda-bad6-1421ed4d8a7b")
    add_experiment(db, TWIN_A, make_runs("bf16", hardware=runpod, experiment_id=TWIN_A))
    add_experiment(
        db, aws_id, make_runs("bf16", hardware=aws, experiment_id=aws_id, latency_scale=1.03)
    )
    same = invoke("compare", "7a8237d0", "95cde129", "--match-by", "cell_key", "--db", db)
    assert same.exit_code == EXIT_OK, same.output
    out = text(same)
    assert "within normal variance (4 load points matched; configs differ in 1 matched" in out
    assert "hardware.cloud runpod → aws" in out
    assert "hardware.instance_type l40s-x1 → g6e.xlarge" in out
    assert "hardware.provider runpod → aws_ec2" in out
    assert "OUTSIDE" not in out

    by_hash = invoke("compare", "7a8237d0", "95cde129", "--db", db)
    assert by_hash.exit_code == EXIT_INVALID, by_hash.output  # nothing pairs by hash

    slow_id = uuid.UUID(int=6)
    add_experiment(
        db, slow_id, make_runs("bf16", hardware=aws, experiment_id=slow_id, latency_scale=1.8)
    )
    slow = invoke("compare", "7a8237d0", str(slow_id), "--match-by", "cell_key", "--db", db)
    assert slow.exit_code == EXIT_MISMATCH, slow.output
    assert "outside normal variance" in text(slow)
