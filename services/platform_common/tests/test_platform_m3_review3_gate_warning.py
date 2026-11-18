"""B3 after the second review (docs/m3-review-fixes-spec.md, "Billing", B3, "Only an outage
counts as unreachable"): refused connections are an outage and are retried; "while the
gate keeps retrying it logs at warning, at most once a minute".

The billing worker starts outside dev with Redis reachable (`TlsRedis`) and Postgres
refusing connections (a `QuietPgProxy` never opened). Its first warning line naming
Postgres is the unreachable-at-start warning. While the gate keeps retrying, at least
one more warning line naming Postgres must follow: within 78 s of the start warning,
which leaves room for a first retry warning logged only after a minute of retrying.

Rate: when that warning comes early (within 30 s), the test watches 20 s more, in which
an unthrottled gate (retrying every few seconds) would log several more; at most once a
minute means none. When it comes late, at most two gate warnings may have been logged.

The gateway's gate is the same code path (`loom_platform.db`); it is not repeated here
to keep the waits of this file down (the test runs 20 times in CI's flake check)."""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from platform_m3_review2_kit import PG_WARNING, QuietPgProxy
from platform_m3_review3_kit import WORKER, RedisFront, redis_front, worker_env
from platform_m3_review_proxy import Service, proxied_url

pytestmark = [pytest.mark.redis]

START_WARN_S = 12
GATE_WARN_S = 78
EARLY_S = 30
WATCH_S = 20


@pytest.fixture
def rf(tmp_path: Path) -> Iterator[RedisFront]:
    with redis_front(tmp_path) as front:
        yield front


def _gate_warnings(svc: Service, start_index: int) -> list[str]:
    """Warning lines naming Postgres after the start warning (output line `start_index`,
    the first such line)."""
    return svc.lines_matching(PG_WARNING, after=start_index + 1)


@pytest.mark.timeout(115)
def test_item4_gate_retrying_refused_connections_logs_a_warning(rf):
    """done-when B3 (outage retried and logged): while the gate keeps retrying a refused
    connection, a warning line naming Postgres is logged (besides the start warning),
    at most once a minute, and the worker keeps running."""
    pg = QuietPgProxy("127.0.0.1", 9)  # never opened: connections refused
    url = proxied_url(
        pg, dbname="loom_review3", user="loom_t_r3_worker", password=secrets.token_hex(16)
    )
    svc = Service(WORKER, worker_env(rf, url))
    try:
        start = svc.wait_line(PG_WARNING, START_WARN_S)
        assert start is not None, f"no unreachable-at-start warning naming Postgres:\n{svc.output}"
        started = time.monotonic()
        first_gate = svc.wait_line(PG_WARNING, GATE_WARN_S, after=start[0] + 1)
        elapsed = time.monotonic() - started
        assert svc.returncode is None, (
            f"the worker exited ({svc.returncode}) during a Postgres outage:\n{svc.output}"
        )
        assert first_gate is not None, (
            f"no warning naming Postgres in {GATE_WARN_S}s of the gate retrying refused "
            f"connections (spec: it logs at warning, at most once a minute):\n{svc.output}"
        )
        if elapsed < EARLY_S:
            time.sleep(WATCH_S)
            gate = _gate_warnings(svc, start[0])
            assert len(gate) == 1, (
                f"{len(gate)} gate warnings within {elapsed + WATCH_S:.0f}s (spec: at most "
                f"once a minute):\n{gate}"
            )
        else:
            gate = _gate_warnings(svc, start[0])
            assert len(gate) <= 2, f"{len(gate)} gate warnings in {elapsed:.0f}s:\n{gate}"
        assert svc.returncode is None, f"the worker exited during the outage:\n{svc.output}"
        assert pg.sessions() == [], "connected although Postgres refused connections"
    finally:
        svc.stop()
        pg.close()
