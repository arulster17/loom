"""The M3 deploy tests' own fakes (`devstack_m3_kit.py`): they must read a correct
`loom-deploy`'s command lines as the real tools would, so a test fails only for what the
spec fixes (§6.12), not for a valid way of calling `aws` or `docker compose`.

- `aws`: a switch (`--no-cli-pager`, `--only-show-errors`, ...) never takes the next word
  as its value, wherever it stands;
- `docker compose`: the project and the files may be given as `-p`/`-f` (absolute or
  relative to the cwd), as COMPOSE_PROJECT_NAME/COMPOSE_FILE in the process env or the env
  file (`--env-file`, else `.env` in the project directory), or (the project) as a
  top-level `name:` in the first compose file, in docker compose's order of precedence;
  `up` and `run` options that take a value (`--wait-timeout`, `--cap-add`, ...) are not
  taken for service names;
- A15's tear-down runs every step (`compose down`, the label sweep, removing the scratch
  root) whatever an earlier one raised, then raises the first error.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from devstack_m3_kit import (
    AWS_BOOL_FLAGS,
    BUCKET,
    Box,
    aws_op,
    compose_calls,
    compose_target,
    deploy_root_base,
    parse_aws_argv,
    runs,
    site,
    ups,
)

SWITCHES = sorted(
    {
        "--no-cli-pager",
        "--no-paginate",
        "--no-sign-request",
        "--no-verify-ssl",
        "--debug",
        "--quiet",
        "--no-progress",
        "--only-show-errors",
        "--with-decryption",
        "--no-with-decryption",
        "--recursive",
        "--dryrun",
        "--no-guess-mime-type",
        "--no-overwrite",
        "--force-glacier-transfer",
        "--ignore-glacier-warnings",
        "--force",
    }
)


@pytest.fixture
def box() -> Iterator[Box]:
    b = Box(deploy_root_base(), site_doc=site())
    yield b
    b.close()


def run_tool(box: Box, tool: str, *args: str, cwd: Path | None = None, **env: str) -> str:
    r = subprocess.run(
        [str(box.bin / tool), *args],
        env=box.env(**env),
        cwd=cwd or box.work,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, f"fake {tool} {args}: exit {r.returncode}\n{r.stderr}"
    return r.stdout


# ---------------------------------------------------------------- aws


def test_the_switches_are_known() -> None:
    assert set(SWITCHES) <= AWS_BOOL_FLAGS


def test_a_global_switch_before_the_service_is_not_a_value() -> None:
    opts, pos = parse_aws_argv(["--no-cli-pager", "ssm", "get-parameter", "--name", "/x"])
    assert pos == ["ssm", "get-parameter"]
    assert opts == {"no-cli-pager": True, "name": "/x"}


def test_a_switch_before_a_path_does_not_swallow_it() -> None:
    opts, pos = parse_aws_argv(["s3", "cp", "--only-show-errors", "s3://b/k", "-"])
    assert pos == ["s3", "cp", "s3://b/k", "-"]
    assert opts == {"only-show-errors": True}


@pytest.mark.parametrize(
    "switch", ["--no-overwrite", "--force-glacier-transfer", "--ignore-glacier-warnings"]
)
def test_the_s3_cp_switches_before_the_paths(switch: str) -> None:
    opts, pos = parse_aws_argv(["s3", "cp", switch, "s3://b/k", "dst"])
    assert pos == ["s3", "cp", "s3://b/k", "dst"]
    assert opts == {switch[2:]: True}


@pytest.mark.parametrize("switch", SWITCHES)
def test_each_switch_anywhere(switch: str) -> None:
    for argv in (
        [switch, "--region", "us-east-1", "s3", "cp", "s3://b/k", "-"],
        ["--region", "us-east-1", "s3", switch, "cp", "s3://b/k", "-"],
        ["--region", "us-east-1", "s3", "cp", switch, "s3://b/k", "-"],
        ["--region", "us-east-1", "s3", "cp", "s3://b/k", switch, "-"],
        ["--region", "us-east-1", "s3", "cp", "s3://b/k", "-", switch],
    ):
        opts, pos = parse_aws_argv(argv)
        assert pos == ["s3", "cp", "s3://b/k", "-"], argv
        assert opts == {"region": "us-east-1", switch[2:]: True}, argv


def test_valued_options_in_both_forms() -> None:
    opts, pos = parse_aws_argv(
        ["--region=eu-west-1", "secretsmanager", "get-secret-value", "--secret-id", "a/b",
         "--query", "SecretString", "--output", "text", "--no-cli-pager"]
    )  # fmt: skip
    assert pos == ["secretsmanager", "get-secret-value"]
    assert opts == {
        "region": "eu-west-1",
        "secret-id": "a/b",
        "query": "SecretString",
        "output": "text",
        "no-cli-pager": True,
    }


def test_the_operation_wherever_the_options_stand() -> None:
    assert aws_op(["ecr", "get-login-password", "--region", "x"]) == ["ecr", "get-login-password"]
    assert aws_op(["--no-cli-pager", "--region", "x", "ecr", "get-login-password"]) == [
        "ecr",
        "get-login-password",
    ]


def test_the_fake_aws_answers_with_switches(box: Box) -> None:
    out = run_tool(
        box, "aws", "--no-cli-pager", "--region", "us-east-1", "ssm", "get-parameter",
        "--name", "/loom/platform/site", "--query", "Parameter.Value", "--output", "text",
    )  # fmt: skip
    assert json.loads(out) == site()
    box.put_object("releases/x.json", b'{"sha": "x"}')
    got = run_tool(
        box, "aws", "s3", "cp", "--only-show-errors", "--no-progress",
        f"s3://{BUCKET}/releases/x.json", "-",
    )  # fmt: skip
    assert got == '{"sha": "x"}'


# ---------------------------------------------------------------- docker compose


def test_the_project_and_files_however_passed() -> None:
    rel = Path("/r/var/lib/loom/releases/s/compose.prod.yml")
    over = Path("/t/override.yml")
    cases = [
        ({"cwd": "/w"}, ["-p", "loom", "-f", str(rel), "-f", str(over)]),
        ({"cwd": "/w"}, ["--project-name=loom", "--file", str(rel), f"--file={over}"]),
        ({"cwd": str(rel.parent), "env": {"COMPOSE_PROJECT_NAME": "loom"}},
         ["-f", "compose.prod.yml", "-f", str(over)]),
        ({"cwd": "/w", "env": {"COMPOSE_PROJECT_NAME": "loom",
                               "COMPOSE_FILE": f"{rel}:{over}"}}, []),
        ({"cwd": str(rel.parent), "env": {"COMPOSE_PROJECT_NAME": "x", "COMPOSE_FILE":
          f"compose.prod.yml;{over}", "COMPOSE_PATH_SEPARATOR": ";"}}, ["-p", "loom"]),
    ]  # fmt: skip
    for record, opts in cases:
        assert compose_target(record, opts) == ("loom", [rel, over]), (record, opts)
    assert compose_target({"cwd": "/w"}, []) == (None, [])


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_the_project_from_the_env_file(tmp_path: Path) -> None:
    compose = write(tmp_path / "rel" / "compose.prod.yml", "services: {}\n")
    env_file = write(tmp_path / "run" / "compose.env", "A=1\nCOMPOSE_PROJECT_NAME=loom\n")
    other = write(tmp_path / "run" / "other.env", "COMPOSE_PROJECT_NAME=x\n")
    write(tmp_path / "rel" / ".env", "COMPOSE_PROJECT_NAME=x\n")  # not read: --env-file given
    cwd = {"cwd": str(tmp_path / "run")}
    for opts in (
        ["-f", str(compose), "--env-file", str(env_file)],
        ["-f", str(compose), f"--env-file={env_file}"],
        ["-f", str(compose), "--env-file", "compose.env"],  # relative to the cwd
        ["-f", str(compose), "--env-file", str(other), "--env-file", str(env_file)],  # the last
    ):
        assert compose_target(cwd, opts) == ("loom", [compose]), opts
    quoted = write(tmp_path / "run" / "quoted.env", "export COMPOSE_PROJECT_NAME='loom'\n")
    assert compose_target(cwd, ["-f", str(compose), "--env-file", str(quoted)])[0] == "loom"


def test_the_project_from_dot_env_in_the_project_directory(tmp_path: Path) -> None:
    compose = write(tmp_path / "rel" / "compose.prod.yml", "services: {}\n")
    write(tmp_path / "rel" / ".env", "COMPOSE_PROJECT_NAME=loom\n")
    write(tmp_path / "work" / ".env", "COMPOSE_PROJECT_NAME=x\n")
    work = {"cwd": str(tmp_path / "work")}
    # The first compose file's directory, not the cwd.
    assert compose_target(work, ["-f", str(compose)]) == ("loom", [compose])
    env = {"COMPOSE_FILE": str(compose)}
    assert compose_target({**work, "env": env}, []) == ("loom", [compose])
    # No compose file named: the cwd.
    assert compose_target({"cwd": str(tmp_path / "rel")}, []) == ("loom", [])
    assert compose_target(work, []) == ("x", [])


def test_the_project_from_the_compose_files_name(tmp_path: Path) -> None:
    over = write(tmp_path / "t" / "override.yml", "services: {}\n")
    cwd = {"cwd": str(tmp_path)}
    for text in (
        "name: loom\nservices: {}\n",
        "# a comment\nservices:\n  web:\n    image: x\nname: loom\n",
        'name: "loom"\n',
        "name: 'loom'  # the project\n",
        "name: loom # the project\n",
    ):
        compose = write(tmp_path / "rel" / "compose.prod.yml", text)
        assert compose_target(cwd, ["-f", str(compose), "-f", str(over)]) == (
            "loom",
            [compose, over],
        ), text
    # Only a top-level key counts.
    compose = write(tmp_path / "rel" / "compose.prod.yml", "services:\n  web:\n    name: x\n")
    assert compose_target(cwd, ["-f", str(compose)]) == (None, [compose])


def test_the_project_precedence(tmp_path: Path) -> None:
    compose = write(tmp_path / "rel" / "compose.prod.yml", "name: from-name\n")
    env_file = write(tmp_path / "run" / "compose.env", "COMPOSE_PROJECT_NAME=from-env-file\n")
    write(tmp_path / "rel" / ".env", "COMPOSE_PROJECT_NAME=from-dot-env\n")
    rec = {"cwd": str(tmp_path)}
    proc = {**rec, "env": {"COMPOSE_PROJECT_NAME": "from-process"}}
    f = ["-f", str(compose)]
    e = ["--env-file", str(env_file)]
    assert compose_target(rec, [*f, *e])[0] == "from-env-file"  # env file beats name:
    assert compose_target(rec, f)[0] == "from-dot-env"  # .env beats name:
    assert compose_target(proc, [*f, *e])[0] == "from-process"  # process env beats env file
    assert compose_target(proc, f)[0] == "from-process"
    for record in (rec, proc):
        for opts in ([*f, *e], f):
            assert compose_target(record, ["-p", "loom", *opts])[0] == "loom", (record, opts)
            assert compose_target(record, [*opts, "--project-name=loom"])[0] == "loom"
    (tmp_path / "rel" / ".env").unlink()
    assert compose_target(rec, f)[0] == "from-name"


def test_a_missing_or_unreadable_env_file_is_skipped(tmp_path: Path) -> None:
    compose = write(tmp_path / "rel" / "compose.prod.yml", "name: loom\n")
    (tmp_path / "a-directory").mkdir()
    rec = {"cwd": str(tmp_path)}
    for env_file in (tmp_path / "missing.env", tmp_path / "a-directory"):
        opts = ["-f", str(compose), "--env-file", str(env_file)]
        assert compose_target(rec, opts) == ("loom", [compose]), env_file
    binary = tmp_path / "binary.env"
    binary.write_bytes(b"\xff\xfe\x00COMPOSE_PROJECT_NAME=x\n")
    assert compose_target(rec, ["-f", str(compose), "--env-file", str(binary)])[0] == "loom"
    missing = tmp_path / "rel" / "missing.yml"
    assert compose_target(rec, ["-f", str(missing)]) == (None, [missing])


def test_compose_file_from_the_env_file(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yml", tmp_path / "t" / "b.yml"
    env_file = write(
        tmp_path / "compose.env", f"COMPOSE_FILE=a.yml;{b}\nCOMPOSE_PATH_SEPARATOR=;\n"
    )
    rec = {"cwd": str(tmp_path)}
    e = ["--env-file", str(env_file)]
    assert compose_target(rec, e) == (None, [a, b])
    proc = {**rec, "env": {"COMPOSE_FILE": str(b)}}
    assert compose_target(proc, e) == (None, [b])  # the process env wins
    assert compose_target(rec, ["-f", str(b), *e]) == (None, [b])  # -f wins


def test_the_fake_docker_and_a_project_in_the_env_file(box: Box) -> None:
    rel = box.var_lib / "releases" / "s"
    rel.mkdir(parents=True)
    env_file = write(box.work / "compose.env", "COMPOSE_PROJECT_NAME=loom\n")
    run_tool(
        box, "docker", "compose", "-f", "compose.prod.yml", "--env-file", str(env_file),
        "pull", cwd=rel,
    )  # fmt: skip
    tl = box.timeline()
    calls = compose_calls(tl)
    assert [c[2] for c in calls] == ["pull"]
    n, opts, _, _ = calls[0]
    assert compose_target(tl[n], opts) == ("loom", [rel / "compose.prod.yml"])


def test_the_fake_docker_records_cwd_and_compose_variables(box: Box) -> None:
    rel = box.var_lib / "releases" / "s"
    rel.mkdir(parents=True)
    run_tool(
        box, "docker", "compose", "-f", "compose.prod.yml", "pull",
        cwd=rel, COMPOSE_PROJECT_NAME="loom",
    )  # fmt: skip
    tl = box.timeline()
    calls = compose_calls(tl)
    assert [c[2] for c in calls] == ["pull"]
    n, opts, _, _ = calls[0]
    assert tl[n]["cwd"] == str(rel)
    assert compose_target(tl[n], opts) == ("loom", [rel / "compose.prod.yml"])


def test_valued_up_and_run_options_are_not_services(box: Box) -> None:
    run_tool(box, "docker", "compose", "-p", "loom", "up", "-d", "--no-deps", "--wait",
             "--wait-timeout", "60", "gateway")  # fmt: skip
    run_tool(box, "docker", "compose", "-p", "loom", "run", "--rm", "--cap-add", "SYS_ADMIN",
             "--cap-drop", "ALL", "-T", "migrate", "loom-migrate")  # fmt: skip
    tl = box.timeline()
    assert [svc for _, svc in ups(tl)] == ["gateway"]
    assert [svc for _, svc, _ in runs(tl)] == ["migrate"]
    assert box.load_docker().get("started") == {"gateway": 1}


# ---------------------------------------------------------------- A15's tear-down


class FakeSh:
    """Stands in for A15's `sh`: records each command and answers the label sweep's
    listings with one id each; it raises `exc` for the commands `fails` picks."""

    def __init__(
        self,
        fails: Callable[[list[str]], bool] = lambda argv: False,
        exc: BaseException | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.fails, self.exc = fails, exc

    def __call__(self, *argv: str, **kw: Any) -> SimpleNamespace:
        self.calls.append(list(argv))
        if self.exc is not None and self.fails(list(argv)):
            raise self.exc
        ids = {"ps": "c1\n", "volume": "v1\n", "network": "n1\n"}
        listing = argv[1] == "ps" or (argv[1] in ("volume", "network") and argv[2] == "ls")
        return SimpleNamespace(returncode=0, stdout=ids[argv[1]] if listing else "", stderr="")

    def ran(self, *prefix: str) -> bool:
        return any(c[: len(prefix)] == list(prefix) for c in self.calls)


@pytest.fixture
def prodstack(monkeypatch: pytest.MonkeyPatch) -> Any:
    import test_devstack_m3_prodstack as mod

    monkeypatch.setattr(mod.shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
    return mod


def is_down(argv: list[str]) -> bool:
    return argv[:2] == ["docker", "compose"] and "down" in argv


def a15_stack(prodstack: Any, box: Box) -> Any:
    override = write(box.work / "override.yml", "services: {}\n")
    write(box.base / "root" / "deep" / "file", "x")
    return prodstack.Stack(box=box, base=box.base, sha="s", override=override)


def test_a15_tear_down_sweeps_and_removes_the_root_after_down_times_out(
    prodstack: Any, box: Box, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeout = subprocess.TimeoutExpired(["docker", "compose", "down"], 600)
    sh = FakeSh(is_down, timeout)
    monkeypatch.setattr(prodstack, "sh", sh)
    with pytest.raises(subprocess.TimeoutExpired) as got:
        prodstack.tear_down(a15_stack(prodstack, box))
    assert got.value is timeout
    assert sh.calls[0][:2] == ["docker", "compose"] and "down" in sh.calls[0]
    assert sh.ran("docker", "ps", "-aq") and sh.ran("docker", "rm", "-f", "c1")
    assert sh.ran("docker", "volume", "ls") and sh.ran("docker", "volume", "rm", "v1")
    assert sh.ran("docker", "network", "ls") and sh.ran("docker", "network", "rm", "n1")
    assert not box.base.exists(), "the scratch root was left behind"


def test_a15_tear_down_goes_on_after_a_sweep_call_fails(
    prodstack: Any, box: Box, monkeypatch: pytest.MonkeyPatch
) -> None:
    boom = subprocess.TimeoutExpired(["docker", "volume", "ls"], 60)
    sh = FakeSh(lambda argv: argv[:3] == ["docker", "volume", "ls"], boom)
    monkeypatch.setattr(prodstack, "sh", sh)
    with pytest.raises(subprocess.TimeoutExpired) as got:
        prodstack.tear_down(a15_stack(prodstack, box))
    assert got.value is boom
    assert sh.ran("docker", "rm", "-f", "c1")
    assert sh.ran("docker", "network", "rm", "n1")
    assert not box.base.exists(), "the scratch root was left behind"


def test_a15_tear_down_runs_the_root_container_fallback_after_errors(
    prodstack: Any, box: Box, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeout = subprocess.TimeoutExpired(["docker", "compose", "down"], 600)
    sh = FakeSh(is_down, timeout)
    monkeypatch.setattr(prodstack, "sh", sh)
    real = prodstack.rmtree
    removals: list[Path] = []

    def rmtree(path: Path) -> None:  # the first pass meets files a root container wrote
        removals.append(path)
        if len(removals) == 1:
            raise PermissionError(f"{path}: root-owned files")
        real(path)

    monkeypatch.setattr(prodstack, "rmtree", rmtree)
    with pytest.raises(subprocess.TimeoutExpired):
        prodstack.tear_down(a15_stack(prodstack, box))
    assert sh.ran("docker", "run", "--rm", "--entrypoint", "rm")
    assert removals == [box.base, box.base]
    assert not box.base.exists(), "the scratch root was left behind"


def test_a15_tear_down_reports_a_root_left_behind(
    prodstack: Any, box: Box, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prodstack, "sh", FakeSh())
    monkeypatch.setattr(prodstack, "rmtree", lambda path: None)
    with pytest.raises(AssertionError, match="scratch root"):
        prodstack.tear_down(a15_stack(prodstack, box))


def test_a15_tear_down_without_errors_raises_nothing(
    prodstack: Any, box: Box, monkeypatch: pytest.MonkeyPatch
) -> None:
    sh = FakeSh()
    monkeypatch.setattr(prodstack, "sh", sh)
    prodstack.tear_down(a15_stack(prodstack, box))
    assert sh.ran("docker", "network", "rm", "n1")
    assert not sh.ran("docker", "run")
    assert not box.base.exists()


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
