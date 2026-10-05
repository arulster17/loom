"""Host shell scripts (rendered from `aws_scripts/*.sh`) and running them over SSM.

Each template starts with `# requires: NAME ...`; `render_script` prepends those
variables as `shlex.quote`d bash assignments (lists become bash arrays) plus the
shared helpers, so values never reach the shell unquoted. Scripts report progress
on stdout as `loom-stage <name> <epoch>` and `loom-sys <key> <value>` lines.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from collections.abc import Awaitable, Callable, Sequence
from importlib.resources import files
from typing import Any

from botocore.exceptions import ClientError  # type: ignore[import-untyped]

ScriptValue = str | int | Sequence[str]

_REQUIRES_RE = re.compile(r"^# requires:(.*)$", re.MULTILINE)
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_HEREDOC_END = "LOOM_SCRIPT_EOF"
TERMINAL_FAILURES = frozenset({"Cancelled", "TimedOut", "Failed", "Cancelling"})
# SSM's own delivery window; the script's run time is bounded by executionTimeout.
DELIVERY_TIMEOUT_S = 600


class SsmCommandError(RuntimeError):
    def __init__(self, command_id: str, status: str, stdout: str, stderr: str) -> None:
        super().__init__(f"SSM command {command_id} {status}: {stderr.strip()[-2000:]}")
        self.command_id = command_id
        self.status = status
        self.stdout = stdout
        self.stderr = stderr


def _template(name: str) -> str:
    return (files("loom_bench.providers") / "aws_scripts" / f"{name}.sh").read_text()


def required_vars(name: str) -> frozenset[str]:
    m = _REQUIRES_RE.search(_template(name))
    if m is None:
        raise ValueError(f"script {name} has no '# requires:' line")
    return frozenset(m.group(1).split())


def _assign(name: str, value: ScriptValue) -> str:
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid script variable {name!r}")
    if isinstance(value, bool):
        raise TypeError(f"{name}: use 0/1, not a bool")
    if isinstance(value, str | int):
        return f"{name}={shlex.quote(str(value))}"
    return f"{name}=({' '.join(shlex.quote(str(v)) for v in value)})"


def render_script(name: str, **values: ScriptValue) -> str:
    """Bash script for template `name` with exactly its required variables set."""
    body = _template(name)
    required = required_vars(name)
    missing, extra = required - values.keys(), values.keys() - required
    if missing or extra:
        raise ValueError(f"script {name}: missing {sorted(missing)}, unexpected {sorted(extra)}")
    lines = ["#!/bin/bash", "set -euo pipefail", "umask 022"]
    lines += [_assign(k, values[k]) for k in sorted(values)]
    script = "\n".join(lines) + "\n\n" + _template("common") + "\n" + body
    if _HEREDOC_END in script:
        raise ValueError(f"script {name} contains the heredoc terminator")
    return script


def ssm_commands(script: str) -> list[str]:
    """AWS-RunShellScript `commands` that run `script` under bash, whatever shell
    the agent uses, and exit with its status."""
    return [
        'f="$(mktemp /tmp/loom.XXXXXX)"',
        f"cat >\"$f\" <<'{_HEREDOC_END}'\n{script}\n{_HEREDOC_END}",
        'bash "$f"; rc=$?; rm -f "$f"; exit "$rc"',
    ]


def parse_markers(stdout: str) -> tuple[dict[str, float], dict[str, str]]:
    """Stage epochs and system facts from a script's stdout. Later lines win."""
    stages: dict[str, float] = {}
    system: dict[str, str] = {}
    for line in stdout.splitlines():
        parts = line.strip().split(" ", 2)
        if len(parts) != 3:
            continue
        kind, key, value = parts
        if kind == "loom-stage":
            try:
                stages[key] = float(value)
            except ValueError:
                continue
        elif kind == "loom-sys":
            system[key] = value
    return stages, system


def stage_offsets(stages: dict[str, float], t0_epoch: float) -> dict[str, float]:
    """Seconds since the stage clock started, in timeline order."""
    ordered = sorted(stages.items(), key=lambda kv: kv[1])
    return {name: round(t - t0_epoch, 3) for name, t in ordered}


async def run_script(
    ssm: Any,
    instance_id: str,
    script: str,
    *,
    timeout_s: int,
    comment: str,
    poll_s: float,
    check_host: Callable[[], Awaitable[None]],
    output_bucket: str | None = None,
    output_prefix: str | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    """Run `script` on the instance and return its stdout.

    `check_host` runs every poll and raises if the host is gone (e.g. a spot
    interruption), so a dead host is reported as such, not as a failed command.
    """
    kwargs: dict[str, Any] = {
        "InstanceIds": [instance_id],
        "DocumentName": "AWS-RunShellScript",
        "Comment": comment[:100],
        "TimeoutSeconds": DELIVERY_TIMEOUT_S,
        "Parameters": {"commands": ssm_commands(script), "executionTimeout": [str(timeout_s)]},
    }
    if output_bucket:
        kwargs["OutputS3BucketName"] = output_bucket
        if output_prefix:
            kwargs["OutputS3KeyPrefix"] = output_prefix
    resp = await asyncio.to_thread(ssm.send_command, **kwargs)
    command_id = resp["Command"]["CommandId"]
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s + DELIVERY_TIMEOUT_S
    while True:
        await sleep(poll_s)
        try:
            inv = await asyncio.to_thread(
                ssm.get_command_invocation, CommandId=command_id, InstanceId=instance_id
            )
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "InvocationDoesNotExist":
                raise
            inv = {"Status": "Pending"}
        status = inv["Status"]
        if status == "Success":
            return str(inv.get("StandardOutputContent", ""))
        if status in TERMINAL_FAILURES:
            await check_host()
            raise SsmCommandError(
                command_id,
                status,
                str(inv.get("StandardOutputContent", "")),
                str(inv.get("StandardErrorContent", "")),
            )
        await check_host()
        if loop.time() > deadline:
            await asyncio.to_thread(
                ssm.cancel_command, CommandId=command_id, InstanceIds=[instance_id]
            )
            raise SsmCommandError(command_id, "ControllerTimeout", "", "")
