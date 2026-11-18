"""python -m loom_gateway: the public API on LOOM_GATEWAY_PORT (8080), Prometheus metrics on
LOOM_GATEWAY_METRICS_PORT (9100), uvloop + httptools."""

from __future__ import annotations

import logging
import socket

import uvicorn
from prometheus_client import start_http_server

from loom_gateway.app import Gateway, create_app
from loom_gateway.logs import configure_logging
from loom_gateway.settings import GatewaySettings
from loom_platform.db import role_gate
from loom_platform.settings import enforce_environment, redis_policy_problems

# After the drain, how long the server still waits for connections to close before it
# cancels what is left (cut streams writing their last chunk; non-streaming requests,
# which then settle `error` and release their reservations).
FINAL_CLOSE_S = 2

log = logging.getLogger("loom_gateway")


class _GateFailedBeforeBind(Exception):
    """Raised from the end of the app's startup to skip uvicorn's bind (`GatewayServer`)."""


class GatewayServer(uvicorn.Server):
    """uvicorn with the gateway's drain between closing the listener and closing the
    connections (contracts §6 "Shutdown"). Plain uvicorn waits for every open stream to
    end (up to LOOM_REQUEST_TIMEOUT_S, 600 s), so the orchestrator's SIGKILL would lose
    their settles: tokens unbilled, reservations and in-flight counts held until the
    reservation TTL (429s for the key meanwhile).

    A failed deferred role check (`loom_platform.db.role_gate`: Postgres was unreachable
    at start) stops the server the same way, and `main` then exits 2. If it has failed
    by the end of the app's startup (the auth listener's wait), the server never binds
    its port: it goes straight to that shutdown (nothing to drain), so a port taken
    meanwhile cannot turn the exit into uvicorn's startup failure (3)."""

    def __init__(self, config: uvicorn.Config, gateway: Gateway, grace_s: float) -> None:
        super().__init__(config)
        self.gateway = gateway
        self.grace_s = grace_s

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        # uvicorn's startup runs the app's lifespan startup, then binds (exit 3 when the
        # bind fails). Check the gate between the two.
        lifespan = self.lifespan
        app_startup = lifespan.startup

        async def app_startup_then_gate() -> None:
            await app_startup()
            if role_gate.failed and not lifespan.should_exit:
                raise _GateFailedBeforeBind

        lifespan.startup = app_startup_then_gate
        try:
            await super().startup(sockets)
        except _GateFailedBeforeBind:
            self.should_exit = True  # `_serve` skips the main loop and its own shutdown
            self.servers = []
            await self.shutdown(sockets)

    async def on_tick(self, counter: int) -> bool:
        if role_gate.failed:
            return True  # as at SIGTERM: drain, settle, close
        return await super().on_tick(counter)

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        for server in self.servers:
            server.close()  # no new connections while draining
        await self.gateway.drain(self.grace_s)
        await super().shutdown(sockets)


def main() -> None:
    settings = GatewaySettings()
    problems = settings.production_problems()
    if settings.env != "dev":
        problems += redis_policy_problems(settings.redis_url)
    enforce_environment(settings, "loom-gateway", problems)
    configure_logging(settings.log_level)
    app = create_app(settings)
    start_http_server(
        settings.gateway_metrics_port,
        addr=settings.gateway_host,
        registry=app.state.gateway.metrics.registry,
    )
    config = uvicorn.Config(
        app,
        host=settings.gateway_host,
        port=settings.gateway_port,
        loop="uvloop",
        http="httptools",
        log_config=None,
        access_log=False,
        timeout_keep_alive=75,
        timeout_graceful_shutdown=FINAL_CLOSE_S,
    )
    try:
        GatewayServer(config, app.state.gateway, settings.shutdown_grace_s).run()
    except BaseException as e:
        # A failed role check decides the exit code, whatever else ended the server
        # (uvicorn's `sys.exit(3)` for a failed bind or app startup, an exception).
        if not role_gate.failed:
            raise
        if isinstance(e, SystemExit):
            log.error("server exited (%s) after the failed database role check", e.code)
        else:
            log.error("server failed after the failed database role check", exc_info=e)
        raise SystemExit(2) from None
    if role_gate.failed:  # the problem is printed; the open streams are settled
        raise SystemExit(2)


if __name__ == "__main__":
    main()
