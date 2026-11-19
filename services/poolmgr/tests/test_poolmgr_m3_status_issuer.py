"""A27, runbook step 4 (docs/phase1-m3-spec.md §7, `docs/runbook-pod-ca.md`): after the
switch to a new CA and `rekey`, the operator checks that every pod's certificate chains
to the new CA: `loom-poolmgr status` shows the issuer of each ready pod's certificate.

A manager on the fakes of `poolmgr_fakes` runs with a pod CA whose common name is the
test's own, and brings two models up; `loom-poolmgr status` then runs as from a shell
(`cli.main` in a thread, its own event loop) against the test's database and Redis, with
the CA and the manager's client certificate in its environment as on the box. Where the
command finds the issuer is its choice; its output names the CA in each ready pod's part:
from the first line naming the pod up to the next line naming another pod or a model.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest
import yaml
from poolmgr_fakes import (
    FakeServing,
    bring_up,
    env_fixture,  # noqa: F401
    serving_config,
    unique_model_id,
)
from poolmgr_m3_kit import CaFiles, model_keys_fixture, write_ca  # noqa: F401

from loom_platform.serving import ServingConfig
from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV, pg_url, redis_client  # noqa: F401
from loom_poolmgr import cli
from loom_poolmgr.pki import Pki

pytestmark = [pytest.mark.postgres, pytest.mark.redis]


def cli_exit(*argv: str) -> int | str | None:
    try:
        cli.main(list(argv))
    except SystemExit as e:
        return e.code
    return 0


def pod_section(out: str, pod_id: str, others: list[str]) -> str:
    """The lines of `out` about `pod_id`: from the first line naming it up to (not
    including) the next line naming another pod or a model in `others`, or the
    month's spend line."""
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if pod_id in line)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if any(o in lines[j] for o in others) or lines[j].startswith("spend this month"):
            end = j
            break
    return "\n".join(lines[start:end])


def status_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: ServingConfig,
    database_url: str,
    ca: CaFiles,
) -> None:
    """Dev, this test's serving file, database, Redis, CA and client certificate."""
    for var in list(os.environ):
        if var.startswith("LOOM_POOLMGR_") or var == "LOOM_SERVING_YAML":
            monkeypatch.delenv(var, raising=False)
    path = tmp_path / "serving.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    monkeypatch.setenv("LOOM_POOLMGR_SERVING_YAML", str(path))
    monkeypatch.setenv("LOOM_ENV", "dev")
    monkeypatch.setenv("LOOM_DATABASE_URL", database_url)
    monkeypatch.setenv("LOOM_REDIS_URL", os.environ.get(REDIS_ENV) or DEFAULT_REDIS)
    monkeypatch.setenv("LOOM_POOLMGR_CA_CERT", str(ca.ca_cert))
    monkeypatch.setenv("LOOM_POOLMGR_CA_KEY", str(ca.ca_key))
    monkeypatch.setenv("LOOM_POOLMGR_CLIENT_CERT", str(ca.client_cert))
    monkeypatch.setenv("LOOM_POOLMGR_CLIENT_KEY", str(ca.client_key))


async def test_status_shows_the_issuer_of_each_ready_pods_certificate(
    env: FakeServing,
    model_keys: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cn = f"Loom m3 status CA {uuid.uuid4().hex[:8]}"
    ca = write_ca(env.root / "ca-status", cn=cn, now=env.clock())
    env.pki = Pki.load(ca.ca_cert, ca.ca_key, ca.client_cert, ca.client_key)
    a, b = unique_model_id(), unique_model_id()
    model_keys += [a, b]
    config = serving_config([a, b])
    manager = env.manager(config)
    await manager.start()
    pods = [await bring_up(env, manager, a), await bring_up(env, manager, b)]
    for model in (a, b):
        assert (await env.pool(model)).routable, model

    status_env(monkeypatch, tmp_path, config, env.database_url, ca)
    capsys.readouterr()
    code = await asyncio.to_thread(cli_exit, "status")
    out = capsys.readouterr().out
    assert code in (0, None), f"loom-poolmgr status exited {code}"
    for pod in pods:
        assert pod.pod_id in out, f"{pod.pod_id} not in the status output:\n{out}"
    ids = [p.pod_id for p in pods]
    for pod in pods:
        others = [i for i in ids if i != pod.pod_id] + [a, b]
        section = pod_section(out, pod.pod_id, others)
        assert cn in section, (
            f"the issuer {cn!r} is not shown for {pod.pod_id}; its section:\n{section}\n"
            f"the whole output:\n{out}"
        )
