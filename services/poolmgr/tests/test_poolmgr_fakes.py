"""The fakes the manager tests stand on (poolmgr_fakes; docs/phase1-m2-spec.md §13: "pods
whose proxy is a real loom-podproxy in front of the mock engine on localhost (real mTLS,
real /metrics)"), driven by hand through the pod exec protocol of loom_poolmgr.bootstrap.

These pass without the manager: a manager test that fails then fails on the manager.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
from pathlib import Path

import pytest
from poolmgr_fakes import (
    FakeClock,
    FakeServingExec,
    FakeServingRunpod,
    ProxyStream,
    make_bundle,
    metric_value,
    proxy_client,
    proxy_complete,
    proxy_get,
    unique_model_id,
)

from loom_bench.providers.runpod_ssh import RemoteCommandError, SshTarget, run_detached
from loom_poolmgr import scripts
from loom_poolmgr.bootstrap import (
    CTL_ROOT,
    ENGINE_UID,
    LEASE_FILE,
    LOOPBACK_ALLOW,
    PROXY_DIR,
    PROXY_UID,
    PROXY_VENV,
    SSH_PROBE,
    UPSTREAM,
    PodSecrets,
    ProxyBundle,
    bootstrap_stdin,
    payload_files,
)
from loom_poolmgr.pki import Pki


def bootstrap_script(bundle: ProxyBundle) -> str:
    return scripts.render(
        "serve_bootstrap",
        PROXY_DIR=PROXY_DIR,
        PROXY_VENV=PROXY_VENV,
        PROXY_PORT=8443,
        UPSTREAM=UPSTREAM,
        BUNDLE_SHA256=bundle.sha256,
        PROXY_UID=PROXY_UID,
        ENGINE_UID=ENGINE_UID,
    )


def engine_script(served: str) -> str:
    values: dict[str, scripts.ScriptValue] = {
        name: "x" for name in scripts.required_vars("serve_engine")
    }
    values.update(SERVED_MODEL=served, ENGINE_STALL_S=60, ENGINE_CMD=["vllm", "serve"])
    return scripts.render("serve_engine", **values)


def lease_script(epoch: int) -> str:
    return scripts.render("serve_lease", LEASE_FILE=LEASE_FILE, LEASE_EPOCH=epoch)


def pod_secrets(pki: Pki, pod_id: str) -> PodSecrets:
    issued = pki.ca.issue_server(pod_id, days=1)
    return PodSecrets(
        server_cert=issued.cert_pem(),
        server_key=issued.key_pem(),
        ca_cert=pki.ca.cert_pem(),
        token=secrets.token_urlsafe(24),
        allow=LOOPBACK_ALLOW,
    )


class Pod:
    """One fake pod created through the fake RunPod API, plus how to reach it."""

    def __init__(self, tmp_path: Path) -> None:
        self.clock = FakeClock()
        self.exec = FakeServingExec(tmp_path / "pods", clock=self.clock)
        self.runpod = FakeServingRunpod(self.exec, clock=self.clock)
        self.api = self.runpod.api(user_agent="loom-poolmgr")
        self.pki = Pki.generate(tmp_path / "pki")
        self.bundle = make_bundle()
        self.model = unique_model_id()

    async def create(self) -> tuple[str, SshTarget]:
        body = {
            "name": f"loom-serve-{self.model}-1790000000-abcdef",
            "ports": ["22/tcp", "8443/tcp"],
            "env": {"LOOM_MANAGED": "true", "LOOM_ROLE": "serving", "LOOM_MODEL": self.model},
        }
        created = await asyncio.to_thread(self.api.create_pod, body)
        pod = await asyncio.to_thread(self.api.get_pod, created["id"])
        assert pod is not None
        assert pod["publicIp"] == "127.0.0.1"
        assert set(pod["portMappings"]) == {"22", "8443"}
        target = SshTarget(
            host="127.0.0.1",
            port=int(pod["portMappings"]["22"]),
            key_path=Path("/nonexistent"),
            known_hosts=Path("/nonexistent"),
        )
        return created["id"], target

    async def bootstrap(self, target: SshTarget, s: PodSecrets) -> str:
        res = await self.exec.run(
            target,
            bootstrap_stdin(bootstrap_script(self.bundle), payload_files(s, self.bundle)),
            timeout_s=60,
        )
        assert res.returncode == 0, res.stderr
        return res.stdout

    async def engine(self, target: SshTarget, *, timeout_s: float = 20) -> str:
        async def check_host() -> None:
            return None

        return await run_detached(
            self.exec,
            target,
            engine_script(self.model),
            what="serve_engine",
            timeout_s=timeout_s,
            poll_s=0.01,
            check_host=check_host,
            ctl_root=CTL_ROOT,
            sleep=asyncio.sleep,
        )

    async def aclose(self) -> None:
        self.api.close()
        await self.exec.aclose()


@pytest.fixture
async def pod(tmp_path: Path):
    p = Pod(tmp_path)
    try:
        yield p
    finally:
        await p.aclose()
    assert not p.exec.protocol_errors, p.exec.protocol_errors


async def test_bootstrap_engine_and_mtls(pod: Pod) -> None:
    pod_id, target = await pod.create()
    assert (await pod.exec.run(target, SSH_PROBE, timeout_s=5)).stdout == "loom-online\n"
    s = pod_secrets(pod.pki, pod_id)
    assert (await pod.bootstrap(target, s)).startswith("loom-proxy started ")
    out = await pod.engine(target)
    assert "loom-stage first_token" in out
    assert "loom-sys engine_isolation ok" in out

    fake = pod.exec.pod(pod_id)
    async with proxy_client(pod.pki) as client:
        replica = fake.replica()
        assert (await proxy_get(client, replica, "/health")).status_code == 200
        assert (await proxy_get(client, replica, "/health", token="wrong")).status_code == 403
        r = await proxy_complete(client, replica, pod.model)
        assert r.status_code == 200, r.text
        metrics = (await proxy_get(client, replica, "/metrics")).text
        assert metric_value(metrics, "vllm:generation_tokens_total") >= 1


async def test_reload_keeps_open_streams_and_swaps_token_and_cert(pod: Pod) -> None:
    pod_id, target = await pod.create()
    first = pod_secrets(pod.pki, pod_id)
    await pod.bootstrap(target, first)
    await pod.engine(target)
    fake = pod.exec.pod(pod_id)
    async with proxy_client(pod.pki) as client:
        old = fake.replica()
        stream = await ProxyStream(client, old, pod.model, max_tokens=60).start()
        second = pod_secrets(pod.pki, pod_id)
        assert (await pod.bootstrap(target, second)).startswith("loom-proxy reloaded ")
        result = await stream.finish()
        assert result.complete and result.completion_tokens == 60, result
        assert (await proxy_get(client, old, "/health")).status_code == 403
        assert (await proxy_get(client, fake.replica(second.token), "/health")).status_code == 200


async def test_bad_bundle_is_refused(pod: Pod) -> None:
    pod_id, target = await pod.create()
    s = pod_secrets(pod.pki, pod_id)
    other = ProxyBundle(b"other", hashlib.sha256(b"other").hexdigest())
    files = payload_files(s, other)
    res = await pod.exec.run(
        target, bootstrap_stdin(bootstrap_script(pod.bundle), files), timeout_s=60
    )
    assert res.returncode == 1
    assert pod.exec.pod(pod_id).proxy is None


async def test_lease_offline_hang_and_terminate(pod: Pod) -> None:
    pod.exec.behaviour.offline_probes = 2
    pod.exec.behaviour.engine = "hang"
    pod_id, target = await pod.create()
    probes = [(await pod.exec.run(target, SSH_PROBE, timeout_s=5)).returncode for _ in range(3)]
    assert probes == [255, 255, 0]
    res = await pod.exec.run(target, lease_script(1790000600), timeout_s=5)
    assert (res.returncode, res.stdout) == (0, "loom-lease 1790000600\n")
    assert pod.exec.pod(pod_id).lease_epoch == 1790000600

    await pod.bootstrap(target, pod_secrets(pod.pki, pod_id))
    with pytest.raises(RemoteCommandError):
        await pod.engine(target, timeout_s=0.3)
    task = asyncio.ensure_future(pod.engine(target))
    await asyncio.sleep(0.2)
    assert not task.done()
    pod.exec.release_engine(pod_id)
    assert "first_token" in await asyncio.wait_for(task, 20)

    await asyncio.to_thread(pod.api.terminate_pod, pod_id)
    assert pod_id in pod.runpod.terminated
    await asyncio.sleep(0.2)
    assert (await pod.exec.run(target, SSH_PROBE, timeout_s=5)).returncode == 255
    async with proxy_client(pod.pki) as client:
        with pytest.raises(Exception):  # noqa: B017 - connection refused, any transport error
            await proxy_get(client, pod.exec.pod(pod_id).replica(), "/health")


async def test_engine_fault_injection(pod: Pod) -> None:
    pod_id, target = await pod.create()
    await pod.bootstrap(target, pod_secrets(pod.pki, pod_id))
    await pod.engine(target)
    fake = pod.exec.pod(pod_id)
    assert fake.engine is not None
    async with proxy_client(pod.pki) as client:
        replica = fake.replica()
        fake.engine.set_healthy(False)
        assert (await proxy_get(client, replica, "/health")).status_code == 503
        fake.engine.set_healthy(True)
        assert (await proxy_get(client, replica, "/health")).status_code == 200

        stream = await ProxyStream(client, replica, pod.model, max_tokens=500).start()
        fake.engine.stall()
        await asyncio.sleep(0.1)
        first = (await proxy_get(client, replica, "/metrics")).text
        await asyncio.sleep(0.3)
        second = (await proxy_get(client, replica, "/metrics")).text
        assert metric_value(second, "vllm:num_requests_running") >= 1
        assert metric_value(first, "vllm:generation_tokens_total") == metric_value(
            second, "vllm:generation_tokens_total"
        )
        fake.engine.kill()
        result = await stream.finish()
        assert not result.complete
        await asyncio.sleep(0.2)
        assert (await proxy_get(client, replica, "/health")).status_code == 503


async def test_fake_clock_sleep_waits_for_advance() -> None:
    clock = FakeClock(start=1000.0)
    task = asyncio.ensure_future(clock.sleep(10))
    await asyncio.sleep(0.01)
    assert not task.done() and clock.next_deadline() == 1010.0
    clock.advance(9)
    await asyncio.sleep(0.01)
    assert not task.done()
    clock.advance(1)
    await asyncio.wait_for(task, 1)
