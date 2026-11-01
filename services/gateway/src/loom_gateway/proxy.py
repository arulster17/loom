"""Dispatch to a replica, stream back, settle exactly once (docs/phase1-contracts.md §6
steps 2-3, §8 steps 5-6, §9, §11, §12; docs/phase1-runpod-plan.md §8).

After a successful reserve every path ends in exactly one `settle` call, which releases
the reservation, charges the engine-reported tokens and appends the usage event:

- complete: the engine's final usage, status `ok`;
- client gone mid-stream: usage of the last chunk forwarded, status `aborted`;
- pod lost mid-stream (or the request timeout): the same, after an `upstream_lost` error
  chunk and `[DONE]`;
- engine 4xx, or no replica answered: nothing billed, status `error`.

A failure before the first byte reaches the client (connect error, TLS failure, 5xx,
stream closed before its first event) is retried once on another replica of the model.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import Any

import httpx
from redis.exceptions import RedisError
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from loom_gateway.metrics import GatewayMetrics
from loom_gateway.request_body import Prepared
from loom_gateway.routing import Router
from loom_gateway.settings import GatewaySettings
from loom_gateway.sse import DONE_EVENT, ChunkRewriter, client_usage, sse_payloads, token_counts
from loom_platform import errors
from loom_platform.auth import AuthRecord
from loom_platform.errors import ApiError
from loom_platform.limiter import Limiter
from loom_platform.pool import PoolState, Replica
from loom_platform.pricing import Price, cost_micros
from loom_platform.usage import UsageEvent, UsageStatus

log = logging.getLogger("loom_gateway.request")

PATHS = {"chat": "/v1/chat/completions", "completions": "/v1/completions"}
_PASS_THROUGH = (400, 413, 422)  # engine statuses that are the client's fault
_BACKGROUND: set[asyncio.Task[Any]] = set()


async def shielded[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run `coro` to completion even if the caller is cancelled (client gone, shutdown)."""
    task = asyncio.ensure_future(coro)
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    return await asyncio.shield(task)


def cost_headers(cost: int) -> dict[str, str]:
    return {
        "X-Loom-Cost-Micros": str(cost),
        "X-Loom-Cost-USD": f"{cost // 1_000_000}.{cost % 1_000_000:06d}",
    }


@dataclass
class RequestCtx:
    request_id: uuid.UUID
    received_at: dt.datetime
    t0: float  # perf_counter at receipt
    record: AuthRecord
    price: Price
    prep: Prepared
    pool: PoolState
    headers: dict[str, str]
    settled: bool = False
    cost: int = 0
    status: str = ""
    t_dispatch: float | None = None

    @property
    def model_id(self) -> str:
        return self.prep.model_id

    @property
    def path(self) -> str:
        return PATHS[self.prep.endpoint]


def _engine_error(status: int, body: bytes) -> ApiError:
    """An engine 4xx as the client's 400, keeping the engine's message."""
    message, param = f"The model server rejected the request ({status}).", None
    try:
        err = json.loads(body).get("error")
        if isinstance(err, dict):
            message = str(err.get("message") or message)
            param = err.get("param") if isinstance(err.get("param"), str) else None
        elif isinstance(err, str):
            message = err
    except (ValueError, AttributeError):
        pass
    low = message.lower()
    code = (
        "context_length_exceeded"
        if "context length" in low or "maximum context" in low
        else "invalid_request"
    )
    return errors.invalid_request(message, param, code)


class Proxy:
    def __init__(
        self,
        settings: GatewaySettings,
        limiter: Limiter,
        router: Router,
        client: httpx.AsyncClient,
        metrics: GatewayMetrics,
    ) -> None:
        self.settings = settings
        self.limiter = limiter
        self.router = router
        self.client = client
        self.metrics = metrics

    # ------------------------------------------------------------ settle

    async def settle(
        self,
        ctx: RequestCtx,
        usage: dict[str, Any] | None,
        status: UsageStatus,
        ttft_s: float | None,
    ) -> int:
        """Charge the request once (later calls return the first result)."""
        if ctx.settled:
            return ctx.cost
        ctx.settled = True
        ctx.status = status
        m = ctx.model_id
        prompt, completion, cached = token_counts(usage)
        if usage is None and status != "error":
            self.metrics.usage_missing.labels(m).inc()
        cost = cost_micros(ctx.price, prompt, completion, cached)
        ctx.cost = cost
        if cost > ctx.prep.max_cost_micros:
            self.metrics.reservation_excess.labels(m).inc()
            log.warning(
                "cost above reservation",
                extra={
                    "fields": {
                        "request_id": str(ctx.request_id),
                        "cost_micros": cost,
                        "max_cost_micros": ctx.prep.max_cost_micros,
                    }
                },
            )
        now = time.perf_counter()
        event = UsageEvent(
            id=ctx.request_id,
            org_id=ctx.record.org_id,
            api_key_id=ctx.record.key_id,
            model_id=m,
            price_version=ctx.price.version,
            cloud=self.settings.cloud,
            region=self.settings.region or None,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_prompt_tokens=cached,
            latency_ms=int((now - ctx.t0) * 1000),
            ttft_ms=None if ttft_s is None else int(ttft_s * 1000),
            status=status,
            cost_micros=cost,
            created_at=ctx.received_at,
        )
        for attempt in range(3):
            try:
                await self.limiter.settle(event)
                break
            except (RedisError, OSError) as e:
                if attempt == 2:
                    self.metrics.settle_failures.inc()
                    log.error(
                        "settle failed: usage not recorded",
                        extra={"fields": {"request_id": str(ctx.request_id), "error": repr(e)}},
                    )
                else:
                    await asyncio.sleep(0.05 * 2**attempt)
        self.metrics.overhead.labels("settle").observe(time.perf_counter() - now)
        self.metrics.inflight.labels(m).dec()
        self.metrics.tokens.labels(m, "prompt").inc(prompt)
        self.metrics.tokens.labels(m, "completion").inc(completion)
        self.metrics.tokens.labels(m, "cached_prompt").inc(cached)
        self.metrics.cost.labels(m).inc(cost)
        self.metrics.duration.labels(ctx.prep.endpoint, m).observe(time.perf_counter() - ctx.t0)
        log.info(
            "request",
            extra={
                "fields": {
                    "request_id": str(ctx.request_id),
                    "org_id": str(ctx.record.org_id),
                    "api_key_id": str(ctx.record.key_id),
                    "endpoint": ctx.prep.endpoint,
                    "model": m,
                    "stream": ctx.prep.stream,
                    "status": status,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "cached_prompt_tokens": cached,
                    "cost_micros": cost,
                    "max_cost_micros": ctx.prep.max_cost_micros,
                    "latency_ms": event.latency_ms,
                    "ttft_ms": event.ttft_ms,
                }
            },
        )
        return cost

    # ------------------------------------------------------------ upstream

    async def _send(self, ctx: RequestCtx, replica: Replica, *, stream: bool) -> httpx.Response:
        if ctx.t_dispatch is None:
            ctx.t_dispatch = time.perf_counter()
            self.metrics.overhead.labels("pre_dispatch").observe(ctx.t_dispatch - ctx.t0)
        request = self.client.build_request(
            "POST",
            replica.url.rstrip("/") + ctx.path,
            content=ctx.prep.upstream_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
                "X-Loom-Pod-Token": replica.token,
                "X-Loom-Request-Id": str(ctx.request_id),
            },
            extensions={"sni_hostname": replica.server_name},
        )
        return await self.client.send(request, stream=stream)

    def _failed(self, ctx: RequestCtx, replica: Replica, why: str) -> None:
        self.router.release(replica, failed=True)
        self.metrics.upstream_failures.labels(ctx.model_id, replica.id).inc()
        log.warning(
            "upstream attempt failed",
            extra={
                "fields": {
                    "request_id": str(ctx.request_id),
                    "model": ctx.model_id,
                    "replica": replica.id,
                    "error": why,
                }
            },
        )

    def _next_replica(self, ctx: RequestCtx, tried: set[str]) -> Replica | None:
        if len(tried) >= 2:
            return None
        replica = self.router.pick(ctx.pool, tried)
        if replica is None:
            return None
        if tried:
            self.metrics.upstream_retries.labels(ctx.model_id).inc()
        tried.add(replica.id)
        self.router.acquire(replica)
        return replica

    async def _fail_request(self, ctx: RequestCtx, error: ApiError) -> Response:
        await shielded(self.settle(ctx, None, "error", None))
        return error_response(error, ctx.headers)

    async def complete(self, ctx: RequestCtx) -> Response:
        """A non-streaming request."""
        tried: set[str] = set()
        while (replica := self._next_replica(ctx, tried)) is not None:
            try:
                resp = await self._send(ctx, replica, stream=False)
            except httpx.HTTPError as e:
                self._failed(ctx, replica, type(e).__name__)
                continue
            if resp.status_code in _PASS_THROUGH:
                self.router.release(replica)
                return await self._fail_request(ctx, _engine_error(resp.status_code, resp.content))
            if resp.status_code != 200:
                self._failed(ctx, replica, f"status {resp.status_code}")
                continue
            try:
                body = resp.json()
                if not isinstance(body, dict):
                    raise ValueError("not an object")
            except ValueError:
                self._failed(ctx, replica, "bad JSON")
                continue
            self.router.release(replica)
            usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
            cost = await shielded(self.settle(ctx, usage, "ok", None))
            if usage is not None:
                body["usage"] = client_usage(usage, cost)
            return Response(
                json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(),
                media_type="application/json",
                headers={**ctx.headers, **cost_headers(cost)},
            )
        return await self._fail_request(ctx, errors.upstream_lost())

    async def stream(self, ctx: RequestCtx) -> Response:
        """A streaming request: commit to a replica once its first event arrives."""
        tried: set[str] = set()
        while (replica := self._next_replica(ctx, tried)) is not None:
            try:
                resp = await self._send(ctx, replica, stream=True)
            except httpx.HTTPError as e:
                self._failed(ctx, replica, type(e).__name__)
                continue
            if resp.status_code != 200:
                try:
                    body = await resp.aread()
                except httpx.HTTPError:
                    body = b""
                await resp.aclose()
                if resp.status_code in _PASS_THROUGH:
                    self.router.release(replica)
                    return await self._fail_request(ctx, _engine_error(resp.status_code, body))
                self._failed(ctx, replica, f"status {resp.status_code}")
                continue
            payloads = sse_payloads(resp.aiter_bytes())
            try:
                first = await anext(payloads)
            except (httpx.HTTPError, StopAsyncIteration) as e:
                await resp.aclose()
                self._failed(ctx, replica, type(e).__name__)
                continue
            return StreamResponse(self, ctx, replica, resp, payloads, first)
        return await self._fail_request(ctx, errors.upstream_lost())


def error_response(error: ApiError, headers: dict[str, str]) -> Response:
    return Response(
        json.dumps(error.body(), separators=(",", ":")).encode(),
        status_code=error.status,
        media_type="application/json",
        headers={**headers, **error.headers},
    )


class StreamResponse(Response):
    """SSE pass-through that notices a client disconnect while waiting on the engine
    (uvicorn's `send` silently drops data after one) and always settles."""

    media_type = "text/event-stream"

    def __init__(
        self,
        proxy: Proxy,
        ctx: RequestCtx,
        replica: Replica,
        upstream: httpx.Response,
        payloads: AsyncIterator[bytes],
        first: bytes,
    ) -> None:
        self.status_code = 200
        self.background = None
        self.proxy = proxy
        self.ctx = ctx
        self.replica = replica
        self.upstream = upstream
        self.payloads = payloads
        self.first = first
        prep = ctx.prep
        self.rewriter = ChunkRewriter(
            include_usage=prep.client_include_usage,
            continuous_usage=prep.client_continuous_usage,
            cost_of=lambda u: cost_micros(ctx.price, *token_counts(u)),
        )
        self.raw_headers = [
            (b"content-type", b"text/event-stream; charset=utf-8"),
            (b"cache-control", b"no-cache"),
            (b"x-accel-buffering", b"no"),
            *((k.lower().encode(), v.encode()) for k, v in ctx.headers.items()),
        ]
        self.last_usage: dict[str, Any] | None = None
        self.saw_done = False
        self.engine_error = False
        self.client_gone = False
        self.lost: str | None = None  # upstream_lost | timeout
        self.first_token_at: float | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        pump: asyncio.Task[None] | None = None
        watch: asyncio.Task[None] | None = None
        try:
            await send({"type": "http.response.start", "status": 200, "headers": self.raw_headers})
            pump = asyncio.create_task(self._pump(send))
            watch = asyncio.create_task(_wait_disconnect(receive))
            await asyncio.wait({pump, watch}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if pump is not None and not pump.done():
                self.client_gone = True
                pump.cancel()
            if watch is not None:
                watch.cancel()
            await asyncio.gather(*(t for t in (pump, watch) if t), return_exceptions=True)
            await shielded(self._finish())

    async def _pump(self, send: Send) -> None:
        remaining = self.ctx.t0 + self.proxy.settings.request_timeout_s - time.perf_counter()
        try:
            async with asyncio.timeout(max(remaining, 0.001)):
                await self._emit(send, self.first)
                if not self.saw_done:
                    async for payload in self.payloads:
                        await self._emit(send, payload)
                        if self.saw_done:
                            break
        except TimeoutError:
            self.lost = "timeout"
        except httpx.HTTPError:
            self.lost = "upstream_lost"
        if not self.saw_done:
            self.lost = self.lost or "upstream_lost"
            await send(
                {
                    "type": "http.response.body",
                    "body": errors.upstream_lost().sse_chunk() + DONE_EVENT,
                    "more_body": True,
                }
            )
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _emit(self, send: Send, payload: bytes) -> None:
        p = self.rewriter.process(payload)
        if p.done:
            # Settle before the client sees [DONE]: the usage event exists once it has.
            self.saw_done = True
            await shielded(self._settle())
            await send({"type": "http.response.body", "body": DONE_EVENT, "more_body": True})
            return
        if p.error:
            self.engine_error = True
        if p.out is not None:
            await send({"type": "http.response.body", "body": p.out, "more_body": True})
        # Recorded only once forwarded: an aborted stream bills what the client was sent.
        if p.usage is not None:
            self.last_usage = p.usage
        if p.content and self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def _status(self) -> UsageStatus:
        if self.engine_error:
            return "error"
        if self.client_gone or self.lost or not self.saw_done:
            return "aborted"
        return "ok"

    async def _settle(self) -> int:
        ttft = None if self.first_token_at is None else self.first_token_at - self.ctx.t0
        return await self.proxy.settle(self.ctx, self.last_usage, self._status(), ttft)

    async def _finish(self) -> None:
        proxy, ctx, m = self.proxy, self.ctx, self.ctx.model_id
        with contextlib.suppress(Exception):
            await self.upstream.aclose()
        proxy.router.release(self.replica, failed=self.lost == "upstream_lost")
        await self._settle()
        if self.client_gone:
            proxy.metrics.aborts.labels(m, "client_gone").inc()
        elif self.lost:
            proxy.metrics.aborts.labels(m, self.lost).inc()
        proxy.metrics.requests.labels(ctx.prep.endpoint, m, "200").inc()
        if self.first_token_at is not None:
            proxy.metrics.ttft.labels(m).observe(self.first_token_at - ctx.t0)
            _, completion, _ = token_counts(self.last_usage)
            elapsed = time.perf_counter() - self.first_token_at
            if completion > 1 and elapsed > 0:
                proxy.metrics.tokens_per_s.labels(m).observe((completion - 1) / elapsed)


async def _wait_disconnect(receive: Receive) -> None:
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return
