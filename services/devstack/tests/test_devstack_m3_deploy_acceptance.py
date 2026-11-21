"""`loom-deploy acceptance --p4-release <sha> [-- <loom-m3-acceptance options>]`: the host's
one half, m3-p's refusal (M3 spec §6.12, §10 m3-p), against the fakes of
`devstack_m3_kit.py`. m3-i's CloudWatch half is not run here (user, 2026-11-19: no Loom
role may read logs, §5.6; B26 runs it by hand): `host.json` is `{"m3-p": {"ok",
"detail"}}`, `ok` a boolean when the host half ran and `null` with detail "not selected"
when it did not (never `true` for a check that did not run); the run exits 0 only when the
container passed and m3-p passed or was not selected; the host half also runs when the
container failed; and no `aws logs` command is called.

Item selection (§6.12, user 2026-11-19): `loom-deploy` reads the options after `--` as
`loom-m3-acceptance` reads them, so the host half runs exactly when the container runs
m3-p. `test_the_host_half_runs_exactly_when_the_driver_runs_m3_p` ties the two together by
parsing the argv the container was given with the driver's own parser and `items()`, not a
second copy of the rule.

Each test first applies release A (fake docker), so the box has a current release, then
publishes release B and runs `acceptance`. The fake docker's `compose run --rm acceptance`
stands in for the container: it writes nothing and exits with a configured code. The fake
`aws` records every call and answers no `logs` operation.

Readings this file takes where §6.12 leaves a choice: the `compose run` of `acceptance`
passes `loom-m3-acceptance` and §6.12's options after the service name, then the options
given after `--`; whether the host half ran is seen in the deploy log, as the
`apply`/`refused` record of `--p4-release` that its `apply --only poolmgr` writes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from devstack_m3_kit import (
    Box,
    aws_op,
    compose_calls,
    compose_target,
    deploy_root_base,
    ecr_images,
    mount_fstype,
    option_values,
    runs,
    secret_values,
    site,
    ups,
)

from loom_devstack import m3_acceptance as m3

SHA_A = "c3" * 20
SHA_B = "d4" * 20


def assert_exit(proc: Any, code: int) -> None:
    """`loom-deploy` ran (an import error also exits 1) and exited with `code`."""
    assert "No module named" not in proc.stderr, ran(proc)
    assert proc.returncode == code, ran(proc)


def ran(proc: Any) -> str:
    out, err = proc.stdout[-3000:], proc.stderr[-3000:]
    return f"exit {proc.returncode}\n--- stdout\n{out}\n--- stderr\n{err}"


@pytest.fixture
def scratch() -> Iterator[Box]:
    b = Box(deploy_root_base(), site_doc=site())
    yield b
    b.close()


@pytest.fixture
def box(scratch: Box) -> Box:
    """Release A applied (the current one), release B published. Built lazily by
    `ready()` in each test, so a missing `loom-deploy` fails the test, not its set-up."""
    return scratch


def ready(b: Box) -> Box:
    b.publish(SHA_A)
    proc = b.deploy("apply", "--release", SHA_A, LOOM_DEPLOY_PROJECT="loom-unit")
    assert proc.returncode == 0, f"applying release A first: {ran(proc)}"
    b.publish(SHA_B, ecr_images(SHA_B))
    b.docker_cfg = b.load_docker()
    return b


def acceptance(box: Box, *args: str) -> Any:
    return box.deploy(
        "acceptance", "--p4-release", SHA_B, *args, LOOM_DEPLOY_PROJECT="loom-unit", timeout=110
    )


def host_json(box: Box) -> dict[str, Any]:
    """§6.12: `{"m3-p": {"ok", "detail"}}`, the one host half; no `m3-i` (B26's). `ok` is
    a boolean when the half ran, `null` with detail "not selected" when it did not."""
    p = box.var_lib / "acceptance" / "host.json"
    assert p.is_file(), "no <root>/var/lib/loom/acceptance/host.json"
    doc: dict[str, Any] = json.loads(p.read_text())
    assert "m3-i" not in doc, f"host.json still has an m3-i half: {doc}"
    assert set(doc) == {"m3-p"}, f"host.json holds more than m3-p: {sorted(doc)}"
    m3p = doc["m3-p"]
    assert isinstance(m3p, dict) and set(m3p) >= {"ok", "detail"}, doc
    assert isinstance(m3p["detail"], str), doc
    ok = m3p["ok"]
    if ok is None:
        assert m3p["detail"] == "not selected", f"ok null means not selected: {doc}"
    else:
        assert isinstance(ok, bool), f"m3-p ok is neither a boolean nor null: {doc}"
        assert m3p["detail"] != "not selected", (
            f"a check that did not run is never true or false, it is null (§6.12): {doc}"
        )
    return doc


NOT_SELECTED = {"m3-p": {"ok": None, "detail": "not selected"}}


def host_half_records(new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The deploy-log records the host half writes: its `apply --only poolmgr --release
    <p4-release>` logs as an apply of release B (refused, §6.12 "Deploy log")."""
    return [r for r in new if r.get("release") == SHA_B or r.get("status") == "refused"]


def assert_host_half_ran(box: Box, new: list[dict[str, Any]]) -> None:
    """The half ran: a boolean `ok`, and its refusal (naming the gateway) in the log."""
    ok = host_json(box)["m3-p"]["ok"]
    assert isinstance(ok, bool), f"m3-p's host half did not run (ok {ok!r}); log: {new}"
    refused = [r for r in new if r.get("step") == "apply" and r.get("status") == "refused"]
    assert len(refused) == 1 and refused[0].get("release") == SHA_B, new
    assert "gateway" in str(refused[0].get("detail")).lower(), refused[0]


def assert_host_half_did_not_run(box: Box, new: list[dict[str, Any]]) -> None:
    """Not selected: `host.json` says so (`ok` null), and no apply of release B is logged."""
    doc = host_json(box)
    assert doc == NOT_SELECTED, f"m3-p not selected, host.json: {doc}"
    assert not host_half_records(new), f"the host half ran though m3-p is not selected: {new}"


def driver_argv(timeline: list[dict[str, Any]]) -> list[str]:
    """The words after `loom-m3-acceptance` in the one `compose run` of `acceptance`."""
    acc = [rest for _, svc, rest in runs(timeline) if svc == "acceptance"]
    assert len(acc) == 1, f"acceptance runs: {acc}"
    rest = acc[0]
    assert "loom-m3-acceptance" in rest, f"the acceptance service runs {rest}"
    return rest[rest.index("loom-m3-acceptance") + 1 :]


def aws_logs_calls(timeline: list[dict[str, Any]]) -> list[list[str]]:
    """The `aws logs ...` calls (any operation) in `timeline`."""
    return [r["argv"] for r in timeline if r["tool"] == "aws" and aws_op(r["argv"])[:1] == ["logs"]]


def test_all_pass_exits_0(box: Box) -> None:
    ready(box)
    n = len(box.timeline())
    proc = acceptance(box, "--", "--only", "m3-0,m3-i,m3-p")
    assert proc.returncode == 0, ran(proc)
    doc = host_json(box)
    assert doc["m3-p"]["ok"] is True, doc
    tl = box.timeline()[n:]
    assert not aws_logs_calls(tl), f"acceptance called aws logs: {aws_logs_calls(tl)}"

    acc = [(i, rest) for i, svc, rest in runs(tl) if svc == "acceptance"]
    assert len(acc) == 1, f"acceptance runs: {acc}"
    rest = acc[0][1]
    after = rest[rest.index("acceptance") + 1 :]
    assert after[:1] == ["loom-m3-acceptance"], f"the acceptance service runs {after[:3]}"
    want = {
        "--domain": site()["domain"],
        "--release": SHA_A,
        "--json": "/var/lib/loom/acceptance/report.json",
        "--deploy-log": "/var/lib/loom/deploy.log",
        "--admin-email": site()["ops_email"],
        "--webhook-secret-file": "/run/loom/secrets/acceptance/webhook_secret",
    }
    for opt, value in want.items():
        assert option_values(after, opt) == [value], f"{opt}: {option_values(after, opt)}"
    assert option_values(after, "--only") == ["m3-0,m3-i,m3-p"], "the options after -- are lost"
    assert not option_values(after, "--canaries-file"), "m3-i has no canaries file (§6.12)"
    for i, opts, _, _ in compose_calls(tl):  # `-p` or COMPOSE_PROJECT_NAME (§6.12)
        assert compose_target(tl[i], opts)[0] == "loom-unit", (opts, tl[i].get("env"))


def test_a_full_run_calls_no_aws_logs_and_host_json_holds_only_m3_p(box: Box) -> None:
    """Every item (no `--only`): m3-i's CloudWatch half is B26's, so no `aws logs` command
    is called and `host.json` has no m3-i (§6.12, §5.6 "Not granted")."""
    ready(box)
    n = len(box.timeline())
    proc = acceptance(box)
    assert proc.returncode == 0, ran(proc)
    calls = aws_logs_calls(box.timeline()[n:])
    assert not calls, f"loom-deploy acceptance called aws logs: {calls}"
    assert host_json(box)["m3-p"]["ok"] is True


def test_a_canaries_file_left_on_the_box_is_not_scanned(box: Box) -> None:
    """A `canaries.json` from an older container (the container writes none now) does not
    bring the CloudWatch scan back: still no `aws logs`, no m3-i in `host.json`, exit 0."""
    ready(box)
    acc = box.var_lib / "acceptance"
    acc.mkdir(parents=True, exist_ok=True)
    (acc / "canaries.json").write_text(json.dumps(["m1canary" + "0123456789abcdef"]))
    n = len(box.timeline())
    proc = acceptance(box)
    assert proc.returncode == 0, ran(proc)
    calls = aws_logs_calls(box.timeline()[n:])
    assert not calls, f"loom-deploy acceptance called aws logs: {calls}"
    assert host_json(box)["m3-p"]["ok"] is True


def test_the_p4_refusal_is_checked_on_the_host(box: Box) -> None:
    """m3-p's host half: `apply --only poolmgr --release <p4-release>` exits 2 naming the
    gateway and the poolmgr container is not recreated (§6.12)."""
    ready(box)
    n_log = len(box.log_records())
    n_tl = len(box.timeline())
    proc = acceptance(box)
    assert proc.returncode == 0, ran(proc)
    assert host_json(box)["m3-p"]["ok"] is True
    new = box.log_records()[n_log:]
    refused = [r for r in new if r["status"] == "refused"]
    assert len(refused) == 1 and refused[0]["release"] == SHA_B, new
    assert "gateway" in str(refused[0].get("detail")).lower(), refused[0]
    assert not [s for _, s in ups(box.timeline()[n_tl:])], "acceptance updated a service"


def test_a_p4_release_that_is_not_refused_fails_m3_p(box: Box) -> None:
    """The current release as `--p4-release`: `--only poolmgr` is then allowed, so the
    refusal m3-p needs does not happen."""
    ready(box)
    n = len(box.timeline())
    proc = box.deploy(
        "acceptance", "--p4-release", SHA_A, LOOM_DEPLOY_PROJECT="loom-unit", timeout=110
    )
    assert_exit(proc, 1)
    assert host_json(box)["m3-p"]["ok"] is False
    calls = aws_logs_calls(box.timeline()[n:])
    assert not calls, f"loom-deploy acceptance called aws logs: {calls}"


def test_a_failing_container_fails_the_run(box: Box) -> None:
    ready(box)
    box.docker_cfg["run_exit"] = {"acceptance": 1}
    box.save_docker()
    n = len(box.timeline())
    proc = acceptance(box)
    assert_exit(proc, 1)
    calls = aws_logs_calls(box.timeline()[n:])
    assert not calls, f"loom-deploy acceptance called aws logs: {calls}"


def test_a_failing_container_still_runs_the_host_half(box: Box) -> None:
    """§6.12: m3-p selected (every item), the container fails: the host half runs anyway
    (its refusal in the deploy log) and the run exits 1. m3-p's `ok` is that half's own
    result: the refusal works, so `true`, whatever the container did."""
    ready(box)
    box.docker_cfg["run_exit"] = {"acceptance": 1}
    box.save_docker()
    n_log = len(box.log_records())
    n_tl = len(box.timeline())
    proc = acceptance(box)
    assert_exit(proc, 1)
    assert_host_half_ran(box, box.log_records()[n_log:])
    doc = host_json(box)
    assert doc["m3-p"]["ok"] is True, f"m3-p's ok is the host half's result, not the run's: {doc}"
    assert not [s for _, s in ups(box.timeline()[n_tl:])], "acceptance updated a service"


def test_only_leaving_m3_p_out_is_not_selected_and_exits_0(box: Box) -> None:
    """`-- --only m3-0`: m3-p is not selected, so the host half does not run: `host.json`
    has m3-p `{"ok": null, "detail": "not selected"}`, no apply of the p4-release is
    logged, and with a passing container the run exits 0 (§6.12)."""
    ready(box)
    n_log = len(box.log_records())
    n_tl = len(box.timeline())
    proc = acceptance(box, "--", "--only", "m3-0")
    assert_exit(proc, 0)
    assert_host_half_did_not_run(box, box.log_records()[n_log:])
    assert option_values(driver_argv(box.timeline()[n_tl:]), "--only") == ["m3-0"]
    assert not [s for _, s in ups(box.timeline()[n_tl:])], "acceptance updated a service"


def test_only_leaving_m3_p_out_with_a_failing_container_exits_1(box: Box) -> None:
    """m3-p not selected and the container fails: still "not selected" (never `true`),
    no host half, and the run exits 1 (it exits 0 only when the container passed)."""
    ready(box)
    box.docker_cfg["run_exit"] = {"acceptance": 1}
    box.save_docker()
    n_log = len(box.log_records())
    proc = acceptance(box, "--", "--only", "m3-0")
    assert_exit(proc, 1)
    assert_host_half_did_not_run(box, box.log_records()[n_log:])


SELECTIONS = [
    pytest.param(["--only", "m3-0", "--only", "m3-p"], id="only-m3-0-then-only-m3-p"),
    pytest.param(["--only", "m3-p", "--only", "m3-0"], id="only-m3-p-then-only-m3-0"),
    pytest.param(["--only=m3-0", "--only", "m3-p"], id="only=m3-0-then-only-m3-p"),
    pytest.param(["--only", "m3-p", "--only=m3-0"], id="only-m3-p-then-only=m3-0"),
    pytest.param(["--only", " m3-0 , m3-p "], id="spaces-around-items"),
    pytest.param(["--external"], id="external"),
    pytest.param([], id="no-options"),
]


@pytest.mark.parametrize("opts", SELECTIONS)
def test_the_host_half_runs_exactly_when_the_driver_runs_m3_p(box: Box, opts: list[str]) -> None:
    """§6.12 "Item selection": `loom-deploy` reads the options after `--` as
    `loom-m3-acceptance` does (the last `--only` wins, `--only=`, spaces ignored,
    `--external` never runs m3-p, no `--only` runs every box item). The items are computed
    from the argv the container was given, with the driver's own parser and `items()`."""
    ready(box)
    n_log = len(box.log_records())
    n_tl = len(box.timeline())
    proc = acceptance(box, "--", *opts) if opts else acceptance(box)
    assert_exit(proc, 0)

    argv = driver_argv(box.timeline()[n_tl:])
    assert argv[len(argv) - len(opts) :] == opts, f"the options after -- are not passed: {argv}"
    driver_items = m3.items(m3.parser().parse_args(argv))
    new = box.log_records()[n_log:]
    if "m3-p" in driver_items:
        assert_host_half_ran(box, new)
        doc = host_json(box)
        assert doc["m3-p"]["ok"] is True, f"the refusal works, so m3-p's ok is true: {doc}"
    else:
        assert_host_half_did_not_run(box, new)


def test_no_secret_on_a_command_line_or_the_output(box: Box) -> None:
    ready(box)
    n = len(box.timeline())
    proc = acceptance(box)
    assert proc.returncode == 0, ran(proc)
    argv = "\n".join(" ".join([r["tool"], *r["argv"]]) for r in box.timeline()[n:])
    for v in [*secret_values(box.store), box.ecr_token]:
        assert v not in argv, "a secret value is on a command line"
        assert v not in proc.stdout + proc.stderr, "acceptance printed a secret value"


def test_refused_when_run_loom_is_not_on_a_tmpfs(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    if mount_fstype(root) == "tmpfs":
        pytest.skip("tmp_path is on a tmpfs here; the refusal needs a disk directory")
    b = Box(deploy_root_base(), root=root, site_doc=site())
    try:
        proc = b.deploy("acceptance", "--p4-release", SHA_B)
        assert_exit(proc, 2)
        assert not [r for r in b.timeline() if r["tool"] == "docker"], "docker ran"
    finally:
        b.close()


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
