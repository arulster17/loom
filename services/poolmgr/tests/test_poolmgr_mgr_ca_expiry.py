"""A CA close to expiry (docs/phase1-m2-spec.md §11 "Fixed after the final whole-branch
review"): the manager refuses to start when the CA expires within
`server_cert_renew_before_s`, and when the CA gets that close while it runs it logs an
error instead of re-keying every pod on every tick.

Pod certificates never outlive their CA (`pki`: not_after is capped at the CA's), so
once the CA is within `server_cert_renew_before_s` of expiry every pod certificate is
"due" for renewal, and renewing cannot help.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from poolmgr_fakes import (
    FakeServing,
    bring_up,
    env_fixture,  # noqa: F401
    serving_config,
    tick,
    unique_model_id,
)

from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr.manager import ManagerLocked
from loom_poolmgr.pki import POOLMGR_CN, CertificateAuthority, Pki, write_issued

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

DAY_S = 86400


def lifecycle(renew_before_s: int) -> dict[str, Any]:
    return {
        "server_cert_days": 7,
        "server_cert_renew_before_s": renew_before_s,
        "lease_ttl_s": 3600,
        "lease_renew_every_s": 300,
    }


def pki_with_ca(directory: Path, *, days: int, now: float) -> Pki:
    """A CA issued at `now` (epoch s) valid `days`, with the manager's client certificate
    (issued at `now` too, capped at the CA's expiry)."""
    directory.mkdir(parents=True, exist_ok=True)
    ca = CertificateAuthority.generate("Loom test CA", days=days, now=now)
    (directory / "ca.crt").write_bytes(ca.cert_pem())
    write_issued(directory, POOLMGR_CN, ca.issue_client(now=now))
    return Pki(
        ca,
        directory / "ca.crt",
        directory / f"{POOLMGR_CN}.crt",
        directory / f"{POOLMGR_CN}.key",
    )


async def test_the_manager_refuses_to_start_on_a_ca_expiring_within_the_renewal_window(
    env: FakeServing, tmp_path: Path
) -> None:
    model = unique_model_id()
    env.pki = pki_with_ca(tmp_path / "ca", days=1, now=env.clock())  # 1 day < 2 days
    manager = env.manager(serving_config([model], lifecycle=lifecycle(2 * DAY_S)))
    with pytest.raises(Exception) as refused:
        await manager.start()
    assert not isinstance(refused.value, ManagerLocked), refused.value
    assert env.runpod.creates() == []


async def test_a_ca_reaching_the_window_while_running_is_logged_not_rekeyed_every_tick(
    env: FakeServing, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = unique_model_id()
    # The CA expires 2 days after the start; the window is 2 days less 10 minutes: outside
    # it at start, inside it after 10 minutes.
    env.pki = pki_with_ca(tmp_path / "ca", days=2, now=env.clock())
    manager = env.manager(serving_config([model], lifecycle=lifecycle(2 * DAY_S - 600)))
    await manager.start()
    pod = await bring_up(env, manager, model)
    delivered = len(pod.payloads)  # the bootstrap's certificate

    with caplog.at_level(logging.ERROR, logger="loom_poolmgr"):
        for _ in range(7):  # 700 s: past the window's edge
            await env.wake(model)
            await tick(manager, env.clock, advance=100)
        crossed = len(pod.payloads)
        for _ in range(5):  # five more ticks inside the window
            await env.wake(model)
            await tick(manager, env.clock, advance=10)

    assert len(pod.payloads) == crossed, (
        f"re-keyed on every tick: {len(pod.payloads) - crossed} deliveries in 5 ticks"
    )
    assert crossed - delivered <= 1, f"{crossed - delivered} re-keys while crossing"
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("CA" in r.getMessage() for r in errors), [r.getMessage() for r in errors]
    assert (await env.pool(model)).routable  # the pod keeps serving
