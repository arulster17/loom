"""The M3 acceptance driver, `loom-m3-acceptance` (M3 spec §10, B10-B19).

It runs on the box (Stage B), so only its command line is checked here: the items of
§10, the two modes (box, and `--external` for what the internet sees: m3-h), and the JSON
report path. The run itself is the Stage B report.
"""

from __future__ import annotations

import contextlib
import io
import sys
from importlib.metadata import entry_points

import pytest

from loom_devstack import m3_acceptance as m3

SCRIPT = "loom-m3-acceptance"
SPEC_ITEMS = {"m3-0", "m3-b", "m3-d", "m3-g", "m3-s", "m3-h", "m3-i", "m3-p", "m3-x"}


def help_text(argv: list[str]) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as exit_:
        m3.main(argv)
    assert exit_.value.code in (0, None)
    return out.getvalue()


def test_the_items_are_those_of_section_10() -> None:
    assert set(m3.ITEMS) == SPEC_ITEMS
    assert set(m3.EXTERNAL_ITEMS) == {"m3-h"}
    assert set(m3.BOX_ITEMS) == SPEC_ITEMS - {"m3-h"}
    for item in SPEC_ITEMS:
        assert callable(getattr(m3.M3Acceptance, "item_" + item.replace("-", "_"))), item


def test_help_shows_the_modes_and_the_report() -> None:
    text = help_text(["--help"])
    for flag in ("--external", "--json", "--domain", "--only", "--brand-name"):
        assert flag in text, flag


def test_a_domain_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOOM_DOMAIN", raising=False)
    with pytest.raises(SystemExit) as exit_, contextlib.redirect_stderr(io.StringIO()):
        m3.main(["--external"])
    assert exit_.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--external", "--only", "m3-0"],  # the flow needs the box
        ["--only", "m3-h"],  # exposure is checked from outside
        ["--only", "m3-z"],
    ],
)
def test_items_outside_the_mode_are_refused(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_, contextlib.redirect_stderr(io.StringIO()):
        m3.main(["--domain", "example.test", *argv])
    assert exit_.value.code == 2


def test_the_options_loom_deploy_passes_are_taken() -> None:
    """§6.12: `loom-deploy acceptance` runs `loom-m3-acceptance --domain --release --json
    --deploy-log --admin-email --webhook-secret-file`; the container calls neither
    loom-deploy nor aws, so their options are gone."""
    argv = [
        "--domain",
        "example.test",
        "--release",
        "a" * 40,
        "--json",
        "/var/lib/loom/acceptance/report.json",
        "--deploy-log",
        "/var/lib/loom/deploy.log",
        "--admin-email",
        "ops@example.test",
        "--webhook-secret-file",
        "/run/loom/secrets/acceptance/webhook_secret",
    ]
    args = m3.parser().parse_args(argv)
    assert args.admin_email == "ops@example.test"
    assert args.webhook_secret_file == "/run/loom/secrets/acceptance/webhook_secret"
    options = {o for a in m3.parser()._actions for o in a.option_strings}
    for gone in ("--deploy-cmd", "--p4-release", "--log-group"):
        assert gone not in options, f"{gone} is still an option"


def test_the_report_and_the_canaries_default_to_the_acceptance_directory() -> None:
    args = m3.parser().parse_args(["--domain", "example.test"])
    assert str(args.json) == "/var/lib/loom/acceptance/report.json"
    assert str(args.canaries_file) == "/var/lib/loom/acceptance/canaries.json"
    assert args.deploy_log == "/var/lib/loom/deploy.log"


def test_the_console_script_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    found = [ep for ep in entry_points(group="console_scripts") if ep.name == SCRIPT]
    assert found, f"no `{SCRIPT}` console script is installed (project.scripts)"
    assert found[0].value == "loom_devstack.m3_acceptance:main", found[0].value
    monkeypatch.setattr(sys, "argv", [SCRIPT, "--help"])
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        found[0].load()()
    assert "--external" in out.getvalue()
