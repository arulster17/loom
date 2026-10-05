import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager

import httpx
import pytest
import uvicorn

from loom_bench.mock.config import MockConfig
from loom_bench.mock.server import create_app

FAST = {"time_scale": 0.001}


@contextmanager
def serve(config: MockConfig) -> Iterator[httpx.Client]:
    """Run the mock on a free port in a background thread."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(config), log_level="critical"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started:
        assert time.monotonic() < deadline, "mock server did not start"
        time.sleep(0.005)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            yield client
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


@pytest.fixture(scope="module")
def client() -> Iterator[httpx.Client]:
    with serve(MockConfig(**FAST)) as c:
        yield c


@pytest.fixture
def make_client() -> Iterator[Callable[..., httpx.Client]]:
    """Start extra servers with config overrides; all are stopped after the test."""
    with ExitStack() as stack:

        def factory(**overrides) -> httpx.Client:
            return stack.enter_context(serve(MockConfig(**{**FAST, **overrides})))

        yield factory
