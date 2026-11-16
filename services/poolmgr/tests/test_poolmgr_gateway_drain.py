"""Item 7 through the gateway (docs/phase1-m2-spec.md §13 item 7; §5 "Idle stop and
drain"; §11 CLI `drain`/`stop`): a stream open on a pod when its drain starts completes
and is billed in full.

A real gateway streams from a pod the manager made ready. While the stream is in flight
the pod is drained, by the admin `drain <pod id>` or `stop <model>` (what `loom-poolmgr
drain|stop` queue). The manager takes the pod out of the pool hash at once but terminates
it only once its engine has no running request: the stream ends normally (`[DONE]`, no
error), its usage event is `ok` with completion tokens = max_tokens, and only then is the
pod terminated. (An idle drain cannot start under an open stream: an in-flight request
is demand, §5.)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import httpx
import pytest
from poolmgr_fakes import (
    FakeServing,
    bring_up,
    env_fixture,  # noqa: F401
    serving_config,
    step,
    unique_model_id,
)
from test_poolmgr_gateway_loop import (
    MESSAGES,
    GatewayFactory,
    GatewayUnderTest,
    Tenant,
    loop_manager,
)
from test_poolmgr_gateway_loop import make_gateway as make_gateway

from loom_platform.pool import wake_key
from loom_platform.testing import pg_url, redis_client  # noqa: F401

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

TOKENS = 150  # ~3 s on the mock (20 ms a token)


class FullStream:
    """A streaming chat completion through the gateway of exactly TOKENS tokens, read in
    the background to its end."""

    def __init__(self, gw: GatewayUnderTest, t: Tenant, model: str) -> None:
        self.gw, self.t, self.model = gw, t, model
        self.events: list[dict[str, Any] | str] = []
        self.status: int | None = None
        self.request_id: str | None = None
        self.error: str | None = None
        self.first = asyncio.Event()
        self.task: asyncio.Task[None] | None = None

    @property
    def finished(self) -> bool:
        return self.task is not None and self.task.done()

    async def start(self, timeout_s: float = 15.0) -> FullStream:
        self.task = asyncio.ensure_future(self._run())
        waiter = asyncio.ensure_future(self.first.wait())
        await asyncio.wait({waiter, self.task}, timeout=timeout_s, return_when="FIRST_COMPLETED")
        waiter.cancel()
        assert self.first.is_set(), f"no first token: {self.status} {self.error}"
        return self

    async def finish(self, timeout_s: float = 60.0) -> None:
        assert self.task is not None
        await asyncio.wait_for(asyncio.shield(self.task), timeout_s)

    async def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            with contextlib.suppress(BaseException):
                await self.task

    async def _run(self) -> None:
        body = {
            "model": self.model,
            "messages": MESSAGES,
            "max_tokens": TOKENS,
            "min_tokens": TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        try:
            async with (
                httpx.AsyncClient(timeout=120, trust_env=False) as c,
                c.stream(
                    "POST",
                    f"{self.gw.url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.t.key}"},
                    json=body,
                ) as r,
            ):
                self.status = r.status_code
                self.request_id = r.headers.get("x-loom-request-id")
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    ev: dict[str, Any] | str = data if data == "[DONE]" else json.loads(data)
                    self.events.append(ev)
                    if isinstance(ev, dict) and ev.get("choices"):
                        self.first.set()
        except httpx.HTTPError as e:
            self.error = f"{type(e).__name__}: {e}"


@pytest.mark.parametrize("op", ["drain", "stop"])
async def test_a_stream_open_when_the_drain_starts_completes_and_is_billed_in_full(
    env: FakeServing,
    make_gateway: GatewayFactory,
    op: str,
) -> None:
    model = unique_model_id()
    gw = await make_gateway([model])
    config = serving_config([model], lifecycle={"drain_max_s": 600})
    manager = loop_manager(env, config, gw)
    await manager.start()
    pod = await bring_up(env, manager, model)
    await env.redis.delete(wake_key(model))
    t = await gw.tenant()

    stream = await FullStream(gw, t, model).start()
    try:
        if op == "drain":
            await env.admin({"op": "drain", "pod_id": pod.pod_id})
        else:
            await env.admin({"op": "stop", "model": model})
        await step(manager, env.clock, advance=1)
        pool = await env.pool(model)
        assert pod.pod_id not in [r.id for r in pool.replicas], "still in the pool hash"

        # The drain waits for the stream: the pod lives until the stream is over.
        for _ in range(400):
            if stream.finished:
                break
            assert env.runpod.has(pod.pod_id), "terminated under an open stream"
            await step(manager, env.clock, advance=0.5)
        await stream.finish()
    finally:
        await stream.cancel()

    assert stream.status == 200 and stream.error is None, (stream.status, stream.error)
    assert stream.events and stream.events[-1] == "[DONE]"
    errors = [e for e in stream.events if isinstance(e, dict) and "error" in e]
    assert not errors, errors
    usages = [e["usage"] for e in stream.events if isinstance(e, dict) and e.get("usage")]
    assert usages and usages[-1]["completion_tokens"] == TOKENS, usages[-1:]

    events = await gw.wait_events(1)
    [event] = [e for e in events if e.org_id == t.org_id]
    assert str(event.id) == stream.request_id
    assert event.status == "ok"
    assert event.completion_tokens == TOKENS
    assert event.cost_micros > 0
    assert await gw.reserved(t) == (0, 0)
    assert await gw.usage(t) == event.cost_micros

    # With the stream over, the drain finishes: terminated, and the model is cold.
    for _ in range(40):
        if not env.runpod.has(pod.pod_id):
            break
        await step(manager, env.clock, advance=1)
    assert not env.runpod.has(pod.pod_id), "the drained pod was never terminated"
    await step(manager, env.clock, advance=1)
    assert (await env.pool(model)).state == "cold"
