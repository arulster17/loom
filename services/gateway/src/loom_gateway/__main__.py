"""python -m loom_gateway: the public API on LOOM_GATEWAY_PORT (8080), Prometheus metrics on
LOOM_GATEWAY_METRICS_PORT (9100), uvloop + httptools."""

from __future__ import annotations

import uvicorn
from prometheus_client import start_http_server

from loom_gateway.app import create_app
from loom_gateway.logs import configure_logging
from loom_gateway.settings import GatewaySettings
from loom_platform.settings import enforce_environment


def main() -> None:
    settings = GatewaySettings()
    enforce_environment(settings, "loom-gateway")
    configure_logging(settings.log_level)
    app = create_app(settings)
    start_http_server(
        settings.gateway_metrics_port,
        addr=settings.gateway_host,
        registry=app.state.gateway.metrics.registry,
    )
    uvicorn.run(
        app,
        host=settings.gateway_host,
        port=settings.gateway_port,
        loop="uvloop",
        http="httptools",
        log_config=None,
        access_log=False,
        timeout_keep_alive=75,
    )


if __name__ == "__main__":
    main()
