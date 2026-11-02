"""python -m loom_gateway: the public API on LOOM_GATEWAY_PORT (8080), Prometheus metrics on
LOOM_GATEWAY_METRICS_PORT (9100), uvloop + httptools."""

from __future__ import annotations

import socket

import uvicorn
from prometheus_client import start_http_server

from loom_gateway.app import Gateway, create_app
from loom_gateway.logs import configure_logging
from loom_gateway.settings import GatewaySettings
from loom_platform.settings import enforce_environment, redis_policy_problems

# After the drain, how long the server still waits for connections to close before it
# cancels what is left (cut streams writing their last chunk; non-streaming requests,
# which then settle `error` and release their reservations).
FINAL_CLOSE_S = 2


class GatewayServer(uvicorn.Server):
    """uvicorn with the gateway's drain between closing the listener and closing the
    connections (contracts §6 "Shutdown"). Plain uvicorn waits for every open stream to
    end (up to LOOM_REQUEST_TIMEOUT_S, 600 s), so the orchestrator's SIGKILL would lose
    their settles: tokens unbilled, reservations and in-flight counts held until the
    reservation TTL (429s for the key meanwhile)."""

    def __init__(self, config: uvicorn.Config, gateway: Gateway, grace_s: float) -> None:
        super().__init__(config)
        self.gateway = gateway
        self.grace_s = grace_s

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
    GatewayServer(config, app.state.gateway, settings.shutdown_grace_s).run()


if __name__ == "__main__":
    main()
