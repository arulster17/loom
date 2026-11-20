"""m3-i in the acceptance container (M3 spec §10 m3-i, §6.12, user 2026-11-19): it scans
the containers' logs only and writes **no** canaries file (nothing on the box reads one;
the CloudWatch half is B26's, run by hand with the operator's credentials). It calls no
`aws` (the container has no AWS credentials, §4).

A fake `docker` on PATH answers M1's item i (`ps -aq`, `logs`, `inspect -f {{.Name}}`);
a recording `aws` fails the test if it is called. The acceptance directory is pointed
into the test's tmp directory (and the cwd too), so a file written there is seen.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from loom_devstack import m3_acceptance as m3
from loom_devstack.acceptance import AcceptanceError, ItemResult, Report, Stack

CANARIES = {"m1canary0011223344556677", "loom_sk_" + "b" * 40}

DOCKER = r"""#!__PYTHON__ -IS
import json, sys
argv = sys.argv[1:]
logs = json.load(open(__LOGS__))
if argv[:2] == ["ps", "-aq"]:
    print("\n".join(logs)); sys.exit(0)
if argv[:1] == ["logs"]:
    print(logs.get(argv[1], "")); sys.exit(0)
if argv[:1] == ["inspect"]:
    print("/loom-" + argv[-1] + "-1"); sys.exit(0)
sys.exit(3)
"""

AWS = r"""#!__PYTHON__ -IS
import sys
open(__CALLED__, "a").write(" ".join(sys.argv[1:]) + "\n")
sys.exit(1)
"""


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    b = tmp_path / "bin"
    b.mkdir()
    logs = tmp_path / "logs.json"
    logs.write_text(json.dumps({"gateway": "GET /v1/models 200\n", "billing": "ok\n"}))
    called = tmp_path / "aws-called"
    for name, text in (("docker", DOCKER), ("aws", AWS)):
        p = b / name
        p.write_text(
            text.replace("__PYTHON__", sys.executable)
            .replace("__LOGS__", repr(str(logs)))
            .replace("__CALLED__", repr(str(called)))
        )
        p.chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}:{os.environ.get('PATH', '/usr/bin:/bin')}")
    (tmp_path / "acc").mkdir()
    monkeypatch.setattr(m3, "ACCEPTANCE_DIR", tmp_path / "acc")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def files(tmp: Path) -> set[str]:
    """Every file under `tmp` but the recording `aws`'s own (checked on its own)."""
    return {str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file()} - {"aws-called"}


BOX_CANARIES = Path("/var/lib/loom/acceptance/canaries.json")


def acceptance(tmp: Path) -> m3.M3Acceptance:
    args = m3.parser().parse_args(["--domain", "example.test"])
    acc = object.__new__(m3.M3Acceptance)
    acc.stack = Stack(
        web="https://example.test",
        gateway="https://api.example.test/v1",
        mailpit="",
        database_url="",
        redis_url="",
        project="loom-unit",
        admin_email="",
        gateway_metrics="",
    )
    acc.report = Report()
    acc.report.canaries = set(CANARIES)
    acc.args = args
    return acc


def test_m3_i_scans_the_logs_writes_no_canaries_file_and_calls_no_aws(tools: Path) -> None:
    before, on_box = files(tools), BOX_CANARIES.exists()
    res = ItemResult("m3-i", "logs")
    acceptance(tools).item_m3_i(res)
    assert res.numbers.get("hits") == {}, res.numbers
    assert not (tools / "aws-called").exists(), "m3-i called aws in the container"
    assert files(tools) == before, f"m3-i wrote files: {sorted(files(tools) - before)}"
    assert BOX_CANARIES.exists() == on_box, f"m3-i wrote {BOX_CANARIES}"


@pytest.mark.parametrize("container", ["gateway", "billing"])
def test_a_canary_in_a_container_log_fails_m3_i_and_no_file_is_written(
    tools: Path, container: str
) -> None:
    logs = tools / "logs.json"
    leaked = sorted(CANARIES)[0]
    logs.write_text(
        json.dumps({"gateway": "ok\n", "billing": "ok\n", container: f"x {leaked} x\n"})
    )
    before = files(tools)
    with pytest.raises(AcceptanceError):
        acceptance(tools).item_m3_i(ItemResult("m3-i", "logs"))
    assert not (tools / "aws-called").exists(), "m3-i called aws in the container"
    assert files(tools) == before, f"m3-i wrote files: {sorted(files(tools) - before)}"


def test_a_key_in_a_container_log_fails_m3_i(tools: Path) -> None:
    """Keys are canaries too (§10 m3-i: no API key in any container's logs)."""
    key = next(c for c in CANARIES if c.startswith("loom_sk_"))
    (tools / "logs.json").write_text(json.dumps({"gateway": f"auth {key}\n"}))
    with pytest.raises(AcceptanceError):
        acceptance(tools).item_m3_i(ItemResult("m3-i", "logs"))
