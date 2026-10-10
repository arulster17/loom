"""`bench site export` publishes what `site/config.yaml` pins, never "the newest of each
name"; the deploy build (`--require-pinned`) refuses a snapshot of anything else."""

import json
from datetime import UTC, datetime

import yaml
from site_helpers import SPEC, make_runs, store_runs
from typer.testing import CliRunner

from loom_bench.cli import EXIT_INVALID, app
from loom_bench.provenance import GitInfo
from loom_bench.store.db import session_scope
from loom_bench.store.repo import create_experiment, update_experiment_status


def invoke(*args: object):
    return CliRunner().invoke(app, [str(a) for a in args])


def text(result) -> str:
    return " ".join(result.output.split())


def experiment(url: str, name: str, day: int, runs=()) -> str:
    with session_scope(url) as s:
        exp = create_experiment(
            s,
            name=name,
            spec=SPEC,
            git=GitInfo(),
            budget_micros=None,
            created_at=datetime(2026, 10, day, tzinfo=UTC),
        )
        store_runs(s, exp.id, runs)
        update_experiment_status(s, exp.id, "completed")
        return str(exp.id)


def config(path, pinned) -> str:
    path.write_text(yaml.safe_dump({"publish": {"experiments": list(pinned)}}))
    return str(path)


def ids(snapshot) -> list[str]:
    return json.loads((snapshot / "manifest.json").read_text())["experiment_ids"]


def test_export_uses_the_pinned_list_not_the_newest_run(empty_db, tmp_path):
    """The 70B on 4x L40S: cf4d1614 is the run to publish, the EAGLE3 sweep 55102ddb
    came later. Pinning the first exports the first."""
    runs = make_runs("vllm-bf16", reps=(1, 2), loads=(2.0, 4.0))
    meant = experiment(empty_db, "llama-70b", 1, runs)
    later = experiment(empty_db, "llama-70b", 2, runs)
    cfg = config(tmp_path / "site.yaml", [meant])
    snap = tmp_path / "data"

    result = invoke("site", "export", "--config", cfg, "--out", snap, "--db", empty_db)
    assert result.exit_code == 0, result.output
    assert ids(snap) == [meant]
    assert "Experiments pinned in" in text(result)
    assert "not the experiments pinned" not in text(result)
    build = invoke("site", "build", "--data", snap, "--config", cfg, "--out", tmp_path / "out")
    assert build.exit_code == 0, build.output
    pinned_build = invoke(
        "site",
        "build",
        "--data",
        snap,
        "--config",
        cfg,
        "--out",
        tmp_path / "o2",
        "--require-pinned",
    )
    assert pinned_build.exit_code == 0, pinned_build.output

    # -e picks something else: written for a preview, flagged, refused by the deploy build.
    preview = invoke(
        "site", "export", "-e", later[:8], "--config", cfg, "--out", snap, "--db", empty_db
    )
    assert preview.exit_code == 0, preview.output
    assert ids(snap) == [later]
    assert "these are not the experiments pinned in" in text(preview)
    refused = invoke(
        "site",
        "build",
        "--data",
        snap,
        "--config",
        cfg,
        "--out",
        tmp_path / "o3",
        "--require-pinned",
    )
    assert refused.exit_code == EXIT_INVALID, refused.output
    assert f"in the snapshot but not pinned: {later}" in text(refused)
    assert f"pinned but not in the snapshot: {meant}" in text(refused)


def test_nothing_pinned_exports_an_empty_snapshot(empty_db, tmp_path):
    experiment(empty_db, "a", 1, make_runs("vllm-bf16", reps=(1, 2), loads=(2.0,)))
    cfg = config(tmp_path / "site.yaml", [])
    snap = tmp_path / "data"
    result = invoke("site", "export", "--config", cfg, "--out", snap, "--db", empty_db)
    assert result.exit_code == 0, result.output
    assert ids(snap) == []
    assert "no experiments pinned in" in text(result) and "the snapshot has no results" in text(
        result
    )
    build = invoke(
        "site",
        "build",
        "--data",
        snap,
        "--config",
        cfg,
        "--out",
        tmp_path / "o",
        "--require-pinned",
    )
    assert build.exit_code == 0, build.output


def test_short_ids_are_refused_in_the_pinned_list(empty_db, tmp_path):
    cfg = config(tmp_path / "site.yaml", ["565b8d3f"])
    result = invoke("site", "export", "--config", cfg, "--out", tmp_path / "d", "--db", empty_db)
    assert result.exit_code == EXIT_INVALID
    assert "publish.experiments needs full experiment ids" in text(result)


def test_the_committed_snapshot_matches_the_committed_pins():
    """What the deploy workflow checks on every push (`--require-pinned`)."""
    from loom_bench.site import load_site_config, load_snapshot
    from loom_bench.site.config import DEFAULT_SNAPSHOT_DIR
    from loom_bench.site.snapshot import check_pinned

    check_pinned(
        load_snapshot(DEFAULT_SNAPSHOT_DIR).manifest, load_site_config().publish.experiments
    )
