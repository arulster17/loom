"""The M3 deploy tests' own fakes (`devstack_m3_kit.py`): they must read a correct
`loom-deploy`'s command lines as the real tools would, so a test fails only for what the
spec fixes (§6.12), not for a valid way of calling `aws` or `docker compose`.

- `aws`: a switch (`--no-cli-pager`, `--only-show-errors`, ...) never takes the next word
  as its value, wherever it stands;
- `docker compose`: the project and the files may be given as `-p`/`-f` (absolute or
  relative to the cwd) or as COMPOSE_PROJECT_NAME/COMPOSE_FILE; `up` and `run` options that
  take a value (`--wait-timeout`, `--cap-add`, ...) are not taken for service names.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

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
