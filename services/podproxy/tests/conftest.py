from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from podproxy_testlib import POD, Upstream, serve_asgi, token

from loom_devstack.devcerts import Issued, ensure_dev_certs, issue, make_ca, write_issued
from loom_podproxy.proxy import PodProxy, ProxyConfig, server_ssl_context


@pytest.fixture(scope="session")
def certs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("certs")
    ensure_dev_certs(out, (POD, "mock-pod-2"))
    # A client certificate from a CA the proxy does not trust.
    rogue: Issued = make_ca("Rogue CA")
    write_issued(out, "rogue", issue(rogue, "loom-gateway", server=False), 0o600)
    return out


@pytest.fixture
def upstream() -> Iterator[tuple[Upstream, int]]:
    app = Upstream()
    with serve_asgi(app) as port:
        yield app, port


@pytest.fixture
async def start_proxy(certs: Path) -> AsyncIterator[Callable[..., Any]]:
    """`await start_proxy(upstream_port, **ProxyConfig overrides)` -> (proxy, port)."""
    proxies: list[PodProxy] = []

    async def factory(upstream_port: int, **overrides: Any) -> tuple[PodProxy, int]:
        config = ProxyConfig(
            token=token(certs),
            upstream=f"http://127.0.0.1:{upstream_port}",
            **overrides,
        )
        ctx = server_ssl_context(certs / f"{POD}.crt", certs / f"{POD}.key", certs / "ca.crt")
        proxy = PodProxy(config, ctx)
        port = await proxy.start("127.0.0.1", 0)
        proxies.append(proxy)
        return proxy, port

    yield factory
    for proxy in proxies:
        await proxy.close()
