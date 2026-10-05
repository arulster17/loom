"""Bad input to any `bench` command is a clean message and exit 2, never a traceback."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from loom_bench.cli import EXIT_INVALID, app

from .conftest import QWEN, mock_doc, write_yaml

NOT_AN_ID = "not-an-id"


def invoke(*args: object):
    return CliRunner().invoke(app, [str(a) for a in args])


def assert_invalid(result, *needles: str) -> None:
    assert result.exit_code == EXIT_INVALID, result.output
    assert isinstance(result.exception, SystemExit)  # handled, not an uncaught exception
    assert "Traceback" not in result.output
    text = " ".join(result.output.split())  # rich wraps long lines
    for needle in needles:
        assert needle in text, result.output


def test_plan_refuses_an_unpriced_instance_type(tmp_path, db):
    doc = yaml.safe_load(QWEN.read_text())
    doc["provider"]["instance_type"] = "g9.nonexistent"
    path = write_yaml(tmp_path / "exp.yaml", doc)
    for command in ("plan", "run"):
        assert_invalid(
            invoke(command, path, "--db", db),
            "instance type g9.nonexistent has no price for aws/us-east-1",
            "docs/how-to/add-gpu-type.md",
        )


def test_plan_refuses_an_unpriced_region(tmp_path, db):
    doc = yaml.safe_load(QWEN.read_text())
    doc["provider"]["region"] = "eu-west-9"
    assert_invalid(
        invoke("plan", write_yaml(tmp_path / "exp.yaml", doc), "--db", db),
        "has no price for aws/eu-west-9",
    )


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("name: a\nname: b\n", "duplicate key 'name'"),
        ("name: [unclosed\n", "invalid experiment"),
        ("- just\n- a list\n", "invalid experiment"),
    ],
)
def test_malformed_experiment_yaml(tmp_path, db, text, needle):
    path = tmp_path / "exp.yaml"
    path.write_text(text)
    assert_invalid(invoke("plan", path, "--db", db), needle)


def test_missing_experiment_file(tmp_path, db):
    assert_invalid(invoke("plan", tmp_path / "nope.yaml", "--db", db), "invalid experiment")


def test_unknown_model_cannot_be_planned(tmp_path, db):
    path = write_yaml(tmp_path / "exp.yaml", mock_doc(model="no-such-model"))
    assert_invalid(invoke("plan", path, "--db", db), "cannot plan unit")


@pytest.mark.parametrize(
    "args",
    [
        ("report", "-e", NOT_AN_ID),
        ("competitiveness", "-e", NOT_AN_ID),
        ("compare", NOT_AN_ID, NOT_AN_ID),
        ("export", "csv", "--out", "x.csv", "--experiment", NOT_AN_ID),
    ],
)
def test_malformed_experiment_ids(db, args):
    assert_invalid(invoke(*args, "--db", db), f"not an experiment id: '{NOT_AN_ID}'")


def test_site_export_of_an_unknown_experiment(db, tmp_path):
    missing = "00000000-0000-0000-0000-000000000001"
    result = invoke("site", "export", "-e", missing, "--out", tmp_path / "snap", "--db", db)
    assert_invalid(result, f"no experiment {missing}")


def test_site_build_without_a_snapshot(tmp_path):
    result = invoke("site", "build", "--data", tmp_path / "none", "--out", tmp_path / "out")
    assert_invalid(result, "cannot build the site")


def test_unknown_quality_suite(tmp_path):
    result = invoke(
        "quality", "run", "no-such-suite", "--base-url", "http://127.0.0.1:1/v1", "--model", "m"
    )
    assert_invalid(result, "invalid quality suite no-such-suite")


def test_quality_gate_with_unknown_refs(db):
    assert_invalid(
        invoke("quality", "gate", "--baseline", "a", "--candidate", "b", "--db", db),
        "cannot re-decide the gate",
    )


def test_bad_job_and_mock_config_files(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert_invalid(invoke("job", "run", "--in", bad, "--out", tmp_path / "o.json"), "load job")
    assert_invalid(invoke("quality", "job", "--in", bad, "--out", tmp_path / "o.json"), "eval job")
    assert_invalid(invoke("mock-server", "--config", tmp_path / "none.yaml"), "mock config")


def test_reproduce_unknown_reference(db, tmp_path):
    assert_invalid(
        invoke("reproduce", NOT_AN_ID, "--db", db, "--out", tmp_path), "cannot reproduce"
    )
