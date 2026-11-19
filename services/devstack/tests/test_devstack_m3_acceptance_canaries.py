"""m3-i in the acceptance container (M3 spec §10 m3-i, §6.12): it scans only the
containers' logs and writes the canaries it planted to `--canaries-file` (default
`/var/lib/loom/acceptance/canaries.json`, mode 0600, a JSON array of strings) for the
host's CloudWatch scan. It calls no `aws` (the container has no AWS credentials, §4).

A fake `docker` on PATH answers M1's item i (`ps -aq`, `logs`, `inspect -f {{.Name}}`);
a recording `aws` fails the test if it is called.
"""

from __future__ import annotations

import json
import os
import stat
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
    return tmp_path


def acceptance(tmp: Path) -> m3.M3Acceptance:
    args = m3.parser().parse_args(
        ["--domain", "example.test", "--canaries-file", str(tmp / "acc" / "canaries.json")]
    )
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


def test_m3_i_writes_the_canaries_0600_and_calls_no_aws(tools: Path) -> None:
    res = ItemResult("m3-i", "logs")
    acceptance(tools).item_m3_i(res)
    path = tools / "acc" / "canaries.json"
    assert path.is_file(), "m3-i wrote no canaries file"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600, oct(path.stat().st_mode)
    data = json.loads(path.read_text())
    assert isinstance(data, list) and set(data) == CANARIES, data
    assert not (tools / "aws-called").exists(), "m3-i called aws in the container"
    assert res.numbers.get("hits") == {}, res.numbers


def test_a_canary_in_a_container_log_fails_and_the_file_is_still_written(tools: Path) -> None:
    logs = tools / "logs.json"
    leaked = sorted(CANARIES)[0]
    logs.write_text(json.dumps({"gateway": f"prompt {leaked} leaked\n"}))
    with pytest.raises(AcceptanceError):
        acceptance(tools).item_m3_i(ItemResult("m3-i", "logs"))
    path = tools / "acc" / "canaries.json"
    assert set(json.loads(path.read_text())) == CANARIES
    assert not (tools / "aws-called").exists()
