"""Cold model plus wake, through to a ready replica (docs/phase1-m2-spec.md §13 items 5 and
6; §5 "Create", "Bootstrap" (in parallel across models); §7 pool-state rows `cold`,
`warming`, `ready` and `not_configured`).

Fake: FakeRunpod behind `RunpodServing`, `FakeServingExec` pods whose proxy is a real
`loom-podproxy` in front of the mock engine on localhost, a fake clock. Real Postgres and
Redis.
"""

from __future__ import annotations

import base64
import json
import logging

import pytest
from poolmgr_fakes import (
    FakeServing,
    PodBehaviour,
    bring_up,
    deployment,
    drive,
    dump_tables,
    env_fixture,  # noqa: F401
    metric_value,
    proxy_complete,
    proxy_get,
    serving_config,
    settle,
    start_command,
    step,
    tick,
    unique_model_id,
)

from loom_bench.providers.runpod_ssh import ssh_argv
from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr.bootstrap import PAYLOAD_MARKER
from loom_poolmgr.provider import POD_NAME_RE

ENGINE_STAGES = {"weights_ready", "engine_started", "engine_healthy", "first_token"}


async def test_cold_wake_creates_one_pod_warming_then_ready(env: FakeServing) -> None:
    """Item 5: one create with the §5 body; `warming` with a decreasing `retry_after_s`
    (max(15, cold_start_estimate_s - elapsed)); then `ready` with exactly that replica after
    bootstrap, health and probe; stages recorded."""
    model = unique_model_id()
    config = serving_config(
        {model: {"cold_start_estimate_s": 600}}, health={"bootstrap_timeout_s": 1800}
    )
    manager = env.manager(config)
    await manager.start()
    env.pods.behaviour.engine = "hang"  # hold the pod in bootstrapping
    await env.wake(model)
    await manager.reconcile_once()
    await drive(env.clock, lambda: any(p.engine_scripts for p in env.pods.snapshot()))

    # The create body (§5 "Create").
    (body,) = env.runpod.creates()
    (t_create,) = env.runpod.create_times
    name = POD_NAME_RE.match(body["name"])
    assert name is not None, body["name"]
    assert name.group("model") == model
    assert abs(int(name.group("epoch")) - t_create) <= 1
    assert body["ports"] == ["22/tcp", "8443/tcp"]
    pod_env = body["env"]
    assert pod_env["LOOM_MANAGED"] == "true"
    assert pod_env["LOOM_ROLE"] == "serving"
    assert pod_env["LOOM_MODEL"] == model
    # now + lease_ttl_s
    assert abs(int(pod_env["LOOM_LEASE_EPOCH"]) - (t_create + 600)) <= 1
    assert pod_env["LOOM_SSH_PUBKEY"] == env.ssh.public_key
    # No HF secret: the checkpoints are ungated.
    assert not [k for k in pod_env if "HF" in k.upper() or "TOKEN" in k.upper()]
    assert "RUNPOD_SECRET" not in json.dumps(body)
    # US datacenters with stock only (FakeRunpod lists US-TX-3, US-TX-4 and EUR-IS-2).
    assert sorted(body["dataCenterIds"]) == ["US-TX-3", "US-TX-4"]
    dep = deployment(config, model)
    assert body["gpuTypeIds"] == [dep.gpu_type_id]
    assert body["gpuCount"] == dep.gpu_count
    assert body["containerDiskInGb"] == 80
    assert body["allowedCudaVersions"] == ["13.0"]
    assert body["interruptible"] is False
    assert body["dockerStartCmd"] == [start_command()]

    # Warming, with a decreasing Retry-After counted from the row's creation, down to the
    # 15 s floor.
    (row,) = await env.rows(model)
    created = row["created_at"].timestamp()
    seen = []
    for advance in (0, 20, 600):
        env.clock.advance(advance)
        await manager.reconcile_once()
        pool = await env.pool(model)
        assert pool.state == "warming", pool
        assert pool.replicas == () and pool.reason is None
        elapsed = env.clock() - created
        assert abs(pool.retry_after_s - max(15, 600 - elapsed)) <= 1, (pool, elapsed)
        seen.append(pool.retry_after_s)
    assert seen[0] > seen[1] > seen[2] == 15

    # The engine comes up: health, probe, ready.
    pod = env.pods.only()
    env.pods.release_engine(pod.pod_id)
    await settle(manager, env.clock)
    for _ in range(5):
        await tick(manager, env.clock, advance=1)
        if (await env.pool(model)).routable:
            break
    pool = await env.pool(model)
    assert pool.state == "ready" and pool.retry_after_s == 0 and pool.reason is None
    assert pool.replicas == (pod.replica(),)
    assert len(env.runpod.creates()) == 1

    # The manager's one-token probe reached the engine through the proxy.
    client = env.client()
    metrics = (await proxy_get(client, pool.replicas[0], "/metrics")).text
    assert metric_value(metrics, "vllm:request_success_total") >= 1
    assert (await proxy_complete(client, pool.replicas[0], model)).status_code == 200

    (row,) = await env.rows(model)
    assert row["state"] == "ready"
    assert row["pod_id"] == pod.pod_id and row["pod_name"] == body["name"]
    assert row["ready_at"] is not None
    assert row["public_ip"] == "127.0.0.1"
    assert row["proxy_port"] == pod.proxy_port and row["ssh_port"] == pod.ssh_port
    assert row["gpu_type_id"] == dep.gpu_type_id and row["gpu_count"] == dep.gpu_count
    assert row["data_center"] in ("US-TX-3", "US-TX-4")
    assert row["cost_per_hr_micros"] >= 1_090_000  # costPerHr 1.09 (+ disk)
    assert row["failure"] is None and row["terminated_at"] is None
    assert {"pod_created", *ENGINE_STAGES} <= set(row["stages"]), row["stages"]


async def test_bootstraps_run_in_parallel_across_models(env: FakeServing) -> None:
    """§5: "Bootstraps run in parallel across models": one model's bootstrap held does
    not hold another's; the second model becomes ready while the first is warming."""
    slow, fast = unique_model_id(), unique_model_id()
    manager = env.manager(serving_config([slow, fast]))
    await manager.start()
    env.pods.behaviour_for[slow] = PodBehaviour(engine="hang")
    await env.wake(slow)
    await env.wake(fast)
    for _ in range(90):
        await step(manager, env.clock, advance=1)
        if (await env.pool(fast)).routable:
            break
    assert (await env.pool(fast)).routable, "the second model waited for the first"
    assert (await env.pool(slow)).state == "warming"
    (held,) = env.pods.of(slow)
    env.pods.release_engine(held.pod_id)
    await settle(manager, env.clock)
    for _ in range(5):
        await tick(manager, env.clock, advance=1)
        if (await env.pool(slow)).routable:
            break
    assert (await env.pool(slow)).routable


async def test_bootstrap_order_over_ssh(env: FakeServing) -> None:
    """§5 "Bootstrap": SSH is probed before anything runs on the pod, the lease is renewed
    before the pod is ready, serve_bootstrap delivers before serve_engine starts."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    pod = await bring_up(env, manager, model)
    kinds = [c.kind for c in list(env.pods.calls) if c.pod_id == pod.pod_id and c.kind != "ctl"]
    assert kinds[0] == "probe", kinds
    assert kinds.index("serve_bootstrap") < kinds.index("serve_engine"), kinds
    assert pod.leases, "no lease renewal before ready"
    assert pod.bootstraps == ["started"]


async def test_waits_for_port_mappings_and_sshd(env: FakeServing) -> None:
    """§5 "Bootstrap" step 1: wait for RUNNING and the 22 and 8443 mappings, then for
    sshd (probes that fail first)."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    env.runpod.mapping_delay = 3
    env.pods.behaviour.offline_probes = 3
    pod = await bring_up(env, manager, model)
    probes = [c for c in list(env.pods.calls) if c.pod_id == pod.pod_id and c.kind == "probe"]
    assert len(probes) >= 4
    assert len(env.runpod.creates()) == 1


async def test_secrets_only_on_ssh_stdin(
    env: FakeServing, caplog: pytest.LogCaptureFixture
) -> None:
    """Item 6: the token, the server key and the certificate appear only on SSH stdin:
    never in the create body or env, any rendered command line, a log line, or Postgres."""
    caplog.set_level(logging.DEBUG)
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    pod = await bring_up(env, manager, model)
    assert pod.payloads, "nothing delivered"
    files = pod.payloads[-1]
    token = files["token"].decode().strip()
    assert (await env.pool(model)).replicas[0].token == token  # the search is meaningful

    def pem_lines(pem: bytes) -> list[str]:
        return [ln for ln in pem.decode().splitlines() if ln and not ln.startswith("-----")]

    needles = {"token": [token]}
    needles["server.key"] = pem_lines(files["server.key"])
    needles["server.crt"] = pem_lines(files["server.crt"])
    calls = list(env.pods.calls)
    (boot,) = [c for c in calls if c.kind == "serve_bootstrap"]
    encoded = boot.stdin.split(f"\n{PAYLOAD_MARKER}\n", 1)[1]
    needles["payload"] = [ln for ln in encoded.splitlines() if len(ln) >= 40][:20]
    assert needles["payload"], "no base64 payload on stdin"
    # Positive control: the payload on stdin does carry them.
    assert token.encode() in base64.b64decode("".join(encoded.split()))

    haystacks = {
        "RunPod requests": "\n".join(
            f"{r.method} {r.url} {r.content.decode(errors='replace')}"
            for r in list(env.runpod.requests)
        ),
        "RunPod pod records": json.dumps(env.runpod.snapshot(), default=str),
        "rendered scripts": "\n".join(c.script for c in calls),
        "ssh command lines": "\n".join(
            " ".join(ssh_argv(c.target)) + f" {c.target}" for c in calls
        ),
        "logs": caplog.text
        + "\n".join(f"{r.getMessage()} {r.exc_text or ''}" for r in caplog.records),
        "postgres": await dump_tables(env.database_url),
    }
    for what, values in needles.items():
        for needle in values:
            for where, text in haystacks.items():
                assert needle not in text, f"{what} found in {where}"


async def test_cold_pool_without_demand(env: FakeServing) -> None:
    """§7 last row: no pod and no demand is `cold` with Retry-After cold_start_estimate_s,
    and nothing is created."""
    model = unique_model_id()
    manager = env.manager(serving_config({model: {"cold_start_estimate_s": 75}}))
    await manager.start()
    for _ in range(3):
        await tick(manager, env.clock, advance=5)
    pool = await env.pool(model)
    assert (pool.state, pool.retry_after_s, pool.reason, pool.replicas) == ("cold", 75, None, ())
    assert env.runpod.creates() == []


async def test_servable_model_missing_from_config_is_not_configured(env: FakeServing) -> None:
    """§4 / §7: a servable model missing from serving.yaml is never started; its pool is
    `unavailable` with reason `not_configured` and `unavailable_retry_after_s`, even when
    woken."""
    model, missing = unique_model_id(), unique_model_id()
    manager = env.manager(serving_config([model]), servable=(model, missing))
    await manager.start()
    await env.wake(missing)
    for _ in range(3):
        await tick(manager, env.clock, advance=5)
    pool = await env.pool(missing)
    assert pool.state == "unavailable" and pool.reason == "not_configured"
    assert pool.retry_after_s == 300
    assert env.runpod.creates() == []
    configured = await env.pool(model)
    assert configured.state == "cold" and configured.reason is None
