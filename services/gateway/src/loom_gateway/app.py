"""The public API (docs/phase1-contracts.md §8 request order):

1. body size (413), bearer token format (401), auth record (401 / 403);
2. model (404), request validation (400);
3. pool state: not routable -> 503 `model_unavailable`, no reservation;
4. reserve (402 / 429), loading the org from Postgres and retrying once on `noinit`;
5. dispatch to a replica, retrying once on another only before the first byte;
6. stream through, settle once, one usage event.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import BlockingConnectionPool, Redis
from redis.exceptions import RedisError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response

from loom_bench.registry import ModelSpec, Registry, load_registry
from loom_gateway.authn import Authenticator
from loom_gateway.metrics import GatewayMetrics
from loom_gateway.proxy import (
    SHUTDOWN_SETTLE_S,
    Proxy,
    RequestCtx,
    StreamResponse,
    error_response,
    shielded,
    wait_background,
)
from loom_gateway.request_body import Endpoint, Prepared, model_field, parse_json_object, prepare
from loom_gateway.routing import Router, upstream_client
from loom_gateway.settings import GatewaySettings
from loom_platform import errors
from loom_platform.auth import AuthRecord
from loom_platform.db import open_pool
from loom_platform.errors import ApiError
from loom_platform.limiter import Limiter
from loom_platform.pricing import Price, servable

log = logging.getLogger("loom_gateway")

EXPOSED_HEADERS = [
    "X-Loom-Request-Id",
    "X-Loom-Price-Version",
    "X-Loom-Max-Cost-Micros",
    "X-Loom-Cost-Micros",
    "X-Loom-Cost-USD",
    "Retry-After",
]


def _usd(micros: int) -> str:
    return f"{micros // 1_000_000}.{micros % 1_000_000:06d}"


class Gateway:
    """Process-wide state: connection pools, caches, the limiter and the router."""

    def __init__(
        self,
        settings: GatewaySettings,
        registry: Registry,
        metrics: GatewayMetrics,
        cold_start_s: dict[str, int] | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.metrics = metrics
        self.cold_start_s = cold_start_s
        self.models: dict[str, tuple[ModelSpec, Price]] = {
            m.id: (m, Price.of(m)) for m in servable(registry)
        }
        self._listener: asyncio.Task[None] | None = None
        self._calls = 0  # generate() calls in progress (a stream's ends at its first byte)

    async def start(self) -> None:
        s = self.settings
        # Blocking: the default pool raises MaxConnectionsError under bursts.
        self.redis = Redis(
            connection_pool=BlockingConnectionPool.from_url(
                s.redis_url, max_connections=s.redis_max_connections, timeout=5
            )
        )
        self.pg: AsyncConnectionPool = await open_pool(
            s.database_url, min_size=1, max_size=s.pg_pool_max
        )
        self.limiter = Limiter(self.redis, usage_stream=s.usage_stream)
        self.auth = Authenticator(
            self.redis, self.pg, redis_ttl_s=s.auth_cache_ttl_s, local_ttl_s=s.auth_local_ttl_s
        )
        self.router = Router(self.redis, s, self.metrics, self.cold_start_s)
        self.client: httpx.AsyncClient = upstream_client(s)
        self.proxy = Proxy(s, self.limiter, self.router, self.client, self.metrics)
        if s.auth_listen:
            self._listener = asyncio.create_task(self.auth.listen_forever(s.database_url))
            try:
                await asyncio.wait_for(self.auth.listening.wait(), 5)
            except TimeoutError:
                log.warning("auth listener not ready; revocation relies on the cache TTL")

    async def drain(self, grace_s: float) -> None:
        """SIGTERM (the listener is already closed): give in-flight requests `grace_s` to
        finish on their own, then cut the open streams; each is billed for the tokens it
        streamed (`aborted`) and releases its reservation and in-flight counts now rather
        than at the reservation's TTL. Requests still running after this are cancelled by
        the server and settle `error` (contracts §6 "Shutdown")."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + grace_s
        while loop.time() < deadline:
            if not (self.proxy.live or self._calls):
                break
            await asyncio.sleep(0.05)
        cut = await self.proxy.abort_streams()
        log.warning(
            "shutdown: drained",
            extra={"fields": {"streams_cut": cut, "requests_running": self._calls}},
        )

    async def stop(self) -> None:
        # Lifespan end: nothing may still hold a reservation when Redis closes.
        await self.proxy.abort_streams()
        left = await wait_background(SHUTDOWN_SETTLE_S)
        if left:
            log.error("shutdown: settles still running at close", extra={"fields": {"n": left}})
        if self._listener is not None:
            self._listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener
        await self.client.aclose()
        await self.pg.close()
        await self.redis.aclose()

    # ------------------------------------------------------------ helpers

    def lookup_model(self, model_id: str) -> tuple[ModelSpec, Price]:
        hit = self.models.get(model_id)
        if hit is None:
            raise errors.model_not_found(model_id)
        return hit

    async def read_body(self, request: Request) -> bytes:
        limit = self.settings.max_body_bytes
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > limit:
                    raise errors.request_too_large(limit)
            except ValueError:
                raise errors.invalid_request("Bad Content-Length.") from None
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > limit:
                raise errors.request_too_large(limit)
        return bytes(body)

    async def reserve(self, record: AuthRecord, request_id: uuid.UUID, prep: Prepared) -> None:
        s = self.settings
        kwargs: dict[str, Any] = {
            "org_id": record.org_id,
            "key_id": record.key_id,
            "request_id": request_id,
            "max_cost_micros": prep.max_cost_micros,
            "tpm_estimate": prep.tpm_estimate,
            "limits": record.limits(s),
            "ttl_ms": s.reservation_ttl_ms,
        }
        result = await self.limiter.reserve(**kwargs)
        if result.reason == "noinit":
            async with self.pg.connection() as conn:
                await self.limiter.load_from_postgres(
                    conn, org_id=record.org_id, key_id=record.key_id
                )
            result = await self.limiter.reserve(**kwargs)
        if result.admitted:
            return
        if result.reason == "noinit":
            raise ApiError(
                503,
                "server_error",
                "balance_unavailable",
                "The balance could not be loaded. Retry shortly.",
                headers={"Retry-After": "1"},
            )
        raise errors.from_reserve(result, prep.max_cost_micros)

    def rejected(
        self, e: ApiError, headers: dict[str, str], endpoint: str, model: str, t0: float
    ) -> Response:
        self.metrics.requests.labels(endpoint, model, str(e.status)).inc()
        self.metrics.errors.labels(e.code).inc()
        log.info(
            "rejected",
            extra={
                "fields": {
                    "request_id": headers.get("X-Loom-Request-Id"),
                    "endpoint": endpoint,
                    "model": model,
                    "http_status": e.status,
                    "code": e.code,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                }
            },
        )
        return error_response(e, headers)

    # ------------------------------------------------------------ endpoints

    async def generate(self, request: Request, endpoint: Endpoint) -> Response:
        t0 = time.perf_counter()
        received_at = dt.datetime.now(dt.UTC)
        request_id = uuid.uuid4()
        self._calls += 1
        try:
            return await self._generate(request, endpoint, t0, received_at, request_id)
        finally:
            self._calls -= 1

    async def _generate(
        self,
        request: Request,
        endpoint: Endpoint,
        t0: float,
        received_at: dt.datetime,
        request_id: uuid.UUID,
    ) -> Response:
        headers = {"X-Loom-Request-Id": str(request_id)}
        model_label = "-"
        try:
            body = await self.read_body(request)
            record = await self.auth.authenticate(request.headers.get("authorization"))
            data = parse_json_object(body)
            model_id = model_field(data)
            spec, price = self.lookup_model(model_id)
            model_label = model_id
            headers["X-Loom-Price-Version"] = price.version
            prep = prepare(
                endpoint,
                data,
                body_bytes=len(body),
                model_id=model_id,
                price=price,
                max_context=spec.max_context,
                max_tokens_cap=self.settings.max_tokens_cap,
            )
            # Reserve before the pool check: a cold model is only woken (a GPU start) for
            # a request that has the balance and passes the rate limits (contracts §8).
            await self.reserve(record, request_id, prep)
            try:
                pool = await self.router.pool(model_id)
                await self.router.check_routable(pool)
            except BaseException:
                # Never dispatched: drop the reservation, no usage event (if this fails
                # too, the reservation expires with its TTL).
                with contextlib.suppress(RedisError, OSError):
                    await shielded(self.limiter.release(record.org_id, request_id))
                raise
            headers["X-Loom-Max-Cost-Micros"] = str(prep.max_cost_micros)
        except ApiError as e:
            return self.rejected(e, headers, endpoint, model_label, t0)

        ctx = RequestCtx(
            request_id=request_id,
            received_at=received_at,
            t0=t0,
            record=record,
            price=price,
            prep=prep,
            pool=pool,
            headers=headers,
        )
        self.metrics.inflight.labels(model_id).inc()
        try:
            if prep.stream:
                response = await self.proxy.stream(ctx)
            else:
                response = await self.proxy.complete(ctx)
        except BaseException:
            if not ctx.settled:
                await shielded(self.proxy.settle(ctx, None, "error", None))
            raise
        if not isinstance(response, StreamResponse):
            self.metrics.requests.labels(endpoint, model_id, str(response.status_code)).inc()
            if response.status_code >= 400:
                self.metrics.errors.labels("upstream").inc()
        return response

    async def list_models(self) -> dict[str, Any]:
        data = []
        for spec, price in self.models.values():
            pool = await self.router.pool(spec.id)
            data.append(
                {
                    "id": spec.id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "loom",
                    "display_name": spec.display_name,
                    "context_length": spec.max_context,
                    "quantization": spec.quantization,
                    "status": spec.status,
                    "capabilities": spec.capabilities.model_dump(),
                    "pricing": {
                        "currency": "USD",
                        "price_version": price.version,
                        "input_per_mtok_micros": price.input_per_mtok,
                        "cached_input_per_mtok_micros": price.cached_input_per_mtok,
                        "output_per_mtok_micros": price.output_per_mtok,
                        "input_per_mtok_usd": _usd(price.input_per_mtok),
                        "cached_input_per_mtok_usd": _usd(price.cached_input_per_mtok),
                        "output_per_mtok_usd": _usd(price.output_per_mtok),
                    },
                    "availability": {
                        "state": pool.state if pool.routable or pool.state != "ready" else "cold",
                        "ready_replicas": len(pool.replicas) if pool.state == "ready" else 0,
                        "retry_after_s": None if pool.routable else pool.retry_after_s,
                    },
                }
            )
        return {"object": "list", "data": data}


def create_app(
    settings: GatewaySettings | None = None,
    *,
    registry: Registry | None = None,
    cold_start_s: dict[str, int] | None = None,
) -> FastAPI:
    settings = settings or GatewaySettings()
    registry = registry or load_registry(settings.models_yaml)
    gw = Gateway(settings, registry, GatewayMetrics(), cold_start_s)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await gw.start()
        try:
            yield
        finally:
            await gw.stop()

    app = FastAPI(
        title="Loom API",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.gateway = gw
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=EXPOSED_HEADERS,
        max_age=600,
    )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> Response:
        code = "not_found" if exc.status_code == 404 else "invalid_request"
        e = ApiError(exc.status_code, "invalid_request_error", code, str(exc.detail))
        return error_response(e, {})

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> Response:
        log.exception("unhandled error", exc_info=exc)
        return JSONResponse(
            {
                "error": {
                    "message": "Internal error.",
                    "type": "server_error",
                    "param": None,
                    "code": "internal_error",
                }
            },
            status_code=500,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await gw.generate(request, "chat")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await gw.generate(request, "completions")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return await gw.list_models()

    @app.get("/v1/models/{model_id:path}")
    async def model(model_id: str) -> Response:
        for m in (await gw.list_models())["data"]:
            if m["id"] == model_id:
                return JSONResponse(m)
        return error_response(errors.model_not_found(model_id), {})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> Response:
        try:
            await gw.redis.ping()
        except Exception:
            return JSONResponse({"status": "redis unavailable"}, status_code=503)
        return JSONResponse({"status": "ok"})

    return app
