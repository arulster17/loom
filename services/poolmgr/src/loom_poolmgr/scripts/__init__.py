"""The serving pod's shell scripts (docs/phase1-m2-spec.md §5, §11 "Serve scripts").

- `serve_start`: the pod's start command (RunPod's `dockerStartCmd`, `bash -c`): the
  lease watchdog and sshd;
- `serve_bootstrap`: over SSH (`bash -s`) with the secrets after it on stdin: the two
  uids, the proxy's files, the proxy;
- `serve_engine`: over SSH, detached (`run_detached`), no secrets: weights, engine,
  isolation checks;
- `serve_lease`: over SSH (`bash -s`): an atomic write of the lease file.

`render(name, **values)` has the bench's `render_script` contract
(`loom_bench.providers.aws_ssm`): each template starts with `# requires: NAME ...`, and
exactly those variables must be given. They are prepended as `shlex.quote`d bash
assignments (lists become bash arrays) after `set -euo pipefail`, followed by this
package's own `common.sh` (`stage`, `sysinfo`, `fail`, `fail_log`). No variable ever
carries a secret: secrets reach a pod only on SSH stdin (`loom_poolmgr.bootstrap`).
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from importlib.resources import files

ScriptValue = str | int | Sequence[str]

SCRIPTS = ("serve_start", "serve_bootstrap", "serve_engine", "serve_lease")

_REQUIRES_RE = re.compile(r"^# requires:(.*)$", re.MULTILINE)
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _template(name: str) -> str:
    if name not in (*SCRIPTS, "common"):
        raise ValueError(f"unknown serve script {name!r} (one of {', '.join(SCRIPTS)})")
    # A Windows checkout may hold CRLF; bash would read every \r as part of a word.
    return (files("loom_poolmgr.scripts") / f"{name}.sh").read_text().replace("\r\n", "\n")


def required_vars(name: str) -> frozenset[str]:
    """The variables template `name` requires (its `# requires:` line)."""
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


def render(name: str, **values: ScriptValue) -> str:
    """Bash script for template `name` with exactly its required variables set."""
    body = _template(name)
    required = required_vars(name)
    missing, extra = required - values.keys(), values.keys() - required
    if missing or extra:
        raise ValueError(f"script {name}: missing {sorted(missing)}, unexpected {sorted(extra)}")
    lines = ["#!/bin/bash", "set -euo pipefail", "umask 022"]
    lines += [_assign(k, values[k]) for k in sorted(values)]
    return "\n".join(lines) + "\n\n" + _template("common") + "\n" + body


def script_name(script: str) -> str:
    """Which template a rendered script came from, by its `# requires:` line (each
    template requires a different set). For fakes that answer the manager's SSH calls."""
    m = _REQUIRES_RE.search(script)
    if m is None:
        raise ValueError("not a rendered serve script")
    found = frozenset(m.group(1).split())
    for name in SCRIPTS:
        if required_vars(name) == found:
            return name
    raise ValueError(f"no serve script requires {sorted(found)}")


def script_var(script: str, name: str) -> str | list[str]:
    """A variable as rendered into `script`: a string, or a list for an array."""
    m = re.search(rf"^{re.escape(name)}=(.*)$", script, re.MULTILINE)
    if m is None:
        raise KeyError(name)
    raw = m.group(1)
    if raw.startswith("("):
        # Arrays are rendered on one line, each element quoted.
        return shlex.split(raw[1:-1])
    (value,) = shlex.split(raw)
    return value
