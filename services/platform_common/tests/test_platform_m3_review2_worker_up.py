"""B3, "Postgres unreachable at start", for the billing worker, with Redis reachable
(docs/m3-review-fixes-spec.md, "Billing", B3): outside dev the service starts with a
warning instead of refusing, and stays up (it comes up and waits for Postgres), rather
than exiting for any reason while Postgres is down.

The worker (`python -m loom_billing worker`) gets a production-shaped configuration:
stripe mode, no HTTP API, a random database password, and a TLS Redis URL with a
password. Redis is reachable (`TlsRedis` in front of the test Redis, accepting any AUTH
password), so the worker cannot exit on Redis instead; its streams have fresh names.
Postgres is a `QuietPgProxy` that is never opened: connections are refused."""

from __future__ import annotations

import secrets
import sys
import time
import uuid
from collections.abc import Iterator

import pytest
import redis
from platform_m3_review2_kit import (
    PG_WARNING,
    QuietPgProxy,
    TlsRedis,
    assert_warning_wording,
    make_certs,
    redis_upstream,
)
from platform_m3_review_proxy import Service, proxied_url, service_env

pytestmark = [pytest.mark.redis]

WORKER = [sys.executable, "-m", "loom_billing", "worker"]
STAYS_UP_S = 5.0


@pytest.fixture
def setup(tmp_path) -> Iterator[dict]:
    host, port, db = redis_upstream()
    certs = make_certs(tmp_path)
    front = TlsRedis((host, port), certs)
    pg = QuietPgProxy("127.0.0.1", 9)  # never opened: Postgres unreachable
    tag = uuid.uuid4().hex[:12]
    streams = {
        "LOOM_USAGE_STREAM": f"loom:test:usage:{tag}",
        "LOOM_USAGE_DEAD_STREAM": f"loom:test:usage:{tag}:dead",
        "LOOM_NOTIFY_STREAM": f"loom:test:usage:{tag}:notify",
    }
    redis_url = front.url(db, secrets.token_hex(12), certs.ca)
    try:
        probe = redis.Redis.from_url(redis_url, socket_timeout=5, socket_connect_timeout=5)
        try:
            assert probe.ping(), "the TLS front for Redis does not answer"
        finally:
            probe.close()
        yield {
            "front": front,
            "pg": pg,
            "env": service_env(
                LOOM_ENV="prod",
                LOOM_DATABASE_URL=proxied_url(
                    pg,
                    dbname="loom_review2",
                    user="loom_t_r2_worker",
                    password=secrets.token_hex(16),
                ),
                LOOM_REDIS_URL=redis_url,
                LOOM_BILLING_METRICS_PORT="0",
                LOOM_LOG_LEVEL="INFO",
                **streams,
            ),
            "streams": list(streams.values()),
        }
    finally:
        pg.close()
        front.close()
        plain = redis.Redis(host=host, port=port, db=db)
        try:
            plain.delete(*streams.values())
        finally:
            plain.close()


def test_item4_worker_pg_down_at_start_still_running_after_warning(setup):
    """done-when B3 (unreachable at start, billing worker): with Redis reachable and
    Postgres unreachable, the worker logs the warning (one line naming Postgres, not the
    role problems' wording) and is still running at least 5 s later: no exit 2, and no
    other exit either."""
    connections_before = setup["front"].connections
    svc = Service(WORKER, setup["env"])
    try:
        warned = svc.wait_for(PG_WARNING, 45)
        assert svc.returncode != 2, (
            f"exited 2 while Postgres was unreachable at start:\n{svc.output}"
        )
        assert warned, f"no warning about Postgres being unreachable:\n{svc.output}"
        assert_warning_wording(svc.output)
        warned_at = time.monotonic()
        code = svc.wait_exit(STAYS_UP_S + 1)
        assert code is None, (
            f"exited ({code}) {time.monotonic() - warned_at:.1f}s after its warning, with "
            f"Redis reachable and Postgres down (spec: it starts and stays up):\n{svc.output}"
        )
        assert setup["front"].connections > connections_before, (
            "the worker never reached Redis: the test did not give it a reachable Redis"
        )
        assert setup["pg"].sessions() == [], "connected although Postgres was unreachable"
    finally:
        svc.stop()
