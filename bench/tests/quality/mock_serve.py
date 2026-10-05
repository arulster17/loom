"""Run the mock backend in a background thread for quality tests."""

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn

from loom_bench.mock.config import MockConfig
from loom_bench.mock.server import create_app

MODEL = "mock-model"


@contextmanager
def serve(**overrides) -> Iterator[str]:
    """Run the mock on a free port in a background thread; yields its /v1 base URL."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = MockConfig(models=[MODEL], **overrides)
    server = uvicorn.Server(uvicorn.Config(create_app(config), log_level="critical"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "mock server did not start"
        time.sleep(0.005)
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
