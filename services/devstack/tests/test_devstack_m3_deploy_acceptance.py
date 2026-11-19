"""`loom-deploy acceptance --p4-release <sha> [-- <loom-m3-acceptance options>]`: the host's
one half, m3-p's refusal (M3 spec §6.12, §10 m3-p), against the fakes of
`devstack_m3_kit.py`. m3-i's CloudWatch half is not run here (user, 2026-11-19: no Loom
role may read logs, §5.6; B26 runs it by hand): `host.json` is `{"m3-p": {"ok",
"detail"}}`, the run exits 0 only when the container and m3-p passed, and no `aws logs`
command is called.

Each test first applies release A (fake docker), so the box has a current release, then
publishes release B and runs `acceptance`. The fake docker's `compose run --rm acceptance`
stands in for the container: it writes nothing and exits with a configured code. The fake
`aws` records every call and answers no `logs` operation.

Readings this file takes where §6.12 leaves a choice: the `compose run` of `acceptance`
passes `loom-m3-acceptance` and §6.12's options after the service name, then the options
given after `--`; `host.json`'s `m3-p` carries a boolean `ok`.
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
    """§6.12: `{"m3-p": {"ok", "detail"}}`, the one host half; no `m3-i` (B26's)."""
    p = box.var_lib / "acceptance" / "host.json"
    assert p.is_file(), "no <root>/var/lib/loom/acceptance/host.json"
    doc: dict[str, Any] = json.loads(p.read_text())
    assert isinstance(doc.get("m3-p"), dict) and isinstance(doc["m3-p"].get("ok"), bool), doc
    assert "detail" in doc["m3-p"], doc
    assert "m3-i" not in doc, f"host.json still has an m3-i half: {doc}"
    assert set(doc) == {"m3-p"}, f"host.json holds more than m3-p: {sorted(doc)}"
    return doc


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
