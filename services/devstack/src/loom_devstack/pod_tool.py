"""Reach a local pod's engine inside the poolmgr container (M1 and M2 acceptance).

With `provider: local` (docs/phase1-m2-spec.md §9) each "pod" is a `loom-podproxy`
process in front of a bench mock engine on a loopback port, both subprocesses of the
pool manager in its container. The acceptance pipes this file to `python -` there
(`docker exec -i <poolmgr> python - <command> <pod id>`), as it does the overhead probe:

    metrics <pod id>   print the engine's own Prometheus text (its /metrics on loopback)
    kill <pod id>      SIGKILL the engine (as a GPU process dying): the proxy stays up,
                       so the pod looks alive to RunPod but fails its health checks

The pod is found by its proxy, whose `--cert` lives under `<pods root>/<pod id>/proxy/`,
and its engine by the proxy's `--upstream` port. Standard library only.
"""

from __future__ import annotations

import os
import signal
import sys
import urllib.request
from pathlib import Path


def processes() -> list[tuple[int, list[str]]]:
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            raw = (d / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in raw.split(b"\0") if a]
        if argv:
            out.append((int(d.name), argv))
    return out


def arg(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv[:-1] else None


def engine_port(pod_id: str) -> int:
    for _, argv in processes():
        cert = arg(argv, "--cert") or ""
        if "loom_podproxy.proxy" in argv and f"/{pod_id}/proxy/" in cert:
            upstream = arg(argv, "--upstream") or ""
            return int(upstream.rsplit(":", 1)[1].split("/", 1)[0])
    raise SystemExit(f"pod_tool: no proxy process for pod {pod_id}")


def engine_pid(port: int) -> int:
    for pid, argv in processes():
        if "loom_bench.mock" in argv and arg(argv, "--port") == str(port):
            return pid
    raise SystemExit(f"pod_tool: no engine process on port {port}")


def main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[0] not in ("metrics", "kill"):
        raise SystemExit("usage: pod_tool (metrics|kill) <pod id>")
    command, pod_id = argv
    port = engine_port(pod_id)
    if command == "metrics":
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
            sys.stdout.write(r.read().decode())
        return
    pid = engine_pid(port)
    os.kill(pid, signal.SIGKILL)
    print(f"killed {pid}")


if __name__ == "__main__":
    main(sys.argv[1:])
