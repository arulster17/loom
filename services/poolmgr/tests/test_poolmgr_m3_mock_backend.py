"""A25: the mock backend outside dev (docs/phase1-m3-spec.md §6.8, §11; contracts §2
`LOOM_POOLMGR_MOCK_BACKEND`, §10).

`LOOM_POOLMGR_MOCK_BACKEND=true` outside `LOOM_ENV=dev`:
- permits `provider: local` (still refused with `runpod`);
- permits `LOOM_POOLMGR_ALLOW_CIDRS`, private (RFC 1918) networks only;
- does not require the SSH key, lease bucket or proxy bundle, still requires the CA and
  the manager's client certificate;
- logs a warning at start and exports `loom_poolmgr_mock_backend 1`.

Checked at two levels: the settings' production checks (`production_problems(config)`,
which `loom-poolmgr run` turns into its exit-2 refusal), and `loom-poolmgr run` itself
(a refusal is exit 2 with no manager built; a start reaches the real manager and serves
the gauge on the metrics endpoint), through `poolmgr_m3_kit.PoolmgrRun`.

The refusals are expected from the manager's `PoolmgrSettings.production_problems` (the
settings check §16 names as a mutation target): the settings-level tests call it
directly, and a refusal by `run` must be the one it reports, not another start-up check
(the runpod case accepts any proxy bundle file, so the bundle's sha check cannot be what
refuses it).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest
from poolmgr_m3_kit import (
    PoolmgrRun,
    poolmgr_run_fixture,  # noqa: F401
    sample,
    serving_file,
)

from loom_platform.serving import ServingConfig, load_serving
from loom_platform.testing import pg_url  # noqa: F401
from loom_poolmgr.bootstrap import ProxyBundle
from loom_poolmgr.manager import PoolManager
from loom_poolmgr.settings import PoolmgrSettings

FLAG = "LOOM_POOLMGR_MOCK_BACKEND"
ALLOW = "LOOM_POOLMGR_ALLOW_CIDRS"
CA_VARS = (
    "LOOM_POOLMGR_CA_CERT",
    "LOOM_POOLMGR_CA_KEY",
    "LOOM_POOLMGR_CLIENT_CERT",
    "LOOM_POOLMGR_CLIENT_KEY",
)
# Inside 10/8, 172.16/12 and 192.168/16 (RFC 1918).
PRIVATE = ["10.0.0.0/8", "172.31.4.0/24", "192.168.10.7", "10.1.0.0/24, 172.16.0.0/12"]
# Anything outside them, including ranges Python calls "private" that RFC 1918 does not
# list (192.0.2.0/24 is documentation space) and a range straddling 10/8.
PUBLIC = [
    "8.8.8.0/24",
    "52.0.0.0/8",
    "0.0.0.0/0",
    "10.0.0.0/7",
    "172.32.0.0/16",
    "100.64.0.0/10",
    "192.0.2.0/24",
    "10.1.0.0/24,52.1.2.3",
]


def config_of(path: Path) -> ServingConfig:
    return load_serving(path, check_registry=False)


def problems(h: PoolmgrRun, provider: str = "local") -> list[str]:
    """The manager's production checks for a serving file with `provider`."""
    path = serving_file(h.tmp / "serving", provider=provider)
    h.mp.setenv("LOOM_POOLMGR_SERVING_YAML", str(path))
    return PoolmgrSettings().production_problems(config_of(path))


def everything_runpod_needs(h: PoolmgrRun) -> None:
    key = h.tmp / "ssh" / "id_ed25519"
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text("not read by the settings check\n")
    key.with_name("id_ed25519.pub").write_text("ssh-ed25519 AAAA test\n")
    bundle = h.tmp / "bundle.tar.gz"
    bundle.write_bytes(b"not read by the settings check")
    h.mp.setenv("LOOM_POOLMGR_SSH_KEY", str(key))
    h.mp.setenv("LOOM_POOLMGR_LEASE_BUCKET", "loom-m3-test-leases")
    h.mp.setenv("LOOM_POOLMGR_PROXY_BUNDLE", str(bundle))
    h.mp.setenv("RUNPOD_API_KEY", "rk-m3-test")


def any_bundle_matches_the_lock(h: PoolmgrRun) -> None:
    """The proxy bundle's sha check (against `config/serving-artifacts.lock`, a bundle
    only the network build makes) accepts whatever file `LOOM_POOLMGR_PROXY_BUNDLE`
    names: wherever `run` checks the bundle, it passes, so a refusal is another rule's."""

    def load(cls: type[ProxyBundle], path: Path, sha256: str) -> ProxyBundle:
        data = Path(path).read_bytes()
        return cls(data, hashlib.sha256(data).hexdigest())

    h.mp.setattr(ProxyBundle, "load", classmethod(load))


def refused_for(h: PoolmgrRun, var: str, value: str) -> bool:
    """`var=value` is refused: when the settings load (naming `var`), or as a production
    problem naming it."""
    h.mp.setenv(var, value)
    try:
        found = problems(h)
    except ValueError as e:
        return var in str(e)
    return any(var in p for p in found)


# ------------------------------------------------------------------ settings checks


def test_flag_permits_local_outside_dev_without_ssh_key_bucket_or_bundle(
    poolmgr_run: PoolmgrRun,
) -> None:
    """With the flag, `provider: local` outside dev has no production problem at all,
    though no SSH key, lease bucket or proxy bundle is set (the CA and client
    certificate are)."""
    poolmgr_run.prod()
    poolmgr_run.mp.setenv(FLAG, "true")
    assert problems(poolmgr_run) == []


@pytest.mark.parametrize("flag", [None, "false"])
def test_without_the_flag_local_stays_dev_only(poolmgr_run: PoolmgrRun, flag: str | None) -> None:
    poolmgr_run.prod()
    if flag is not None:
        poolmgr_run.mp.setenv(FLAG, flag)
    found = problems(poolmgr_run)
    assert any("local" in p for p in found), found


def test_flag_is_refused_with_runpod(poolmgr_run: PoolmgrRun) -> None:
    """A runpod setup with everything it needs passes `production_problems` without the
    flag; the same setup with the flag is refused there, naming the mock backend (M4
    removes the flag)."""
    poolmgr_run.prod()
    everything_runpod_needs(poolmgr_run)
    assert problems(poolmgr_run, "runpod") == []
    poolmgr_run.mp.setenv(FLAG, "true")
    found = problems(poolmgr_run, "runpod")
    assert any("mock" in p.lower() for p in found), found


@pytest.mark.parametrize("var", CA_VARS)
def test_flag_still_requires_the_ca_and_the_client_certificate(
    poolmgr_run: PoolmgrRun, var: str
) -> None:
    poolmgr_run.prod()
    poolmgr_run.mp.setenv(FLAG, "true")
    poolmgr_run.mp.delenv(var)
    found = problems(poolmgr_run)
    assert any(var in p for p in found), found


@pytest.mark.parametrize("cidrs", PRIVATE)
def test_flag_permits_private_allow_cidrs(poolmgr_run: PoolmgrRun, cidrs: str) -> None:
    poolmgr_run.prod()
    poolmgr_run.mp.setenv(FLAG, "true")
    assert not refused_for(poolmgr_run, ALLOW, cidrs)
    assert problems(poolmgr_run) == []


@pytest.mark.parametrize("cidrs", PUBLIC)
def test_flag_refuses_non_rfc1918_allow_cidrs(poolmgr_run: PoolmgrRun, cidrs: str) -> None:
    poolmgr_run.prod()
    poolmgr_run.mp.setenv(FLAG, "true")
    assert refused_for(poolmgr_run, ALLOW, cidrs)


def test_allow_cidrs_outside_dev_without_the_flag_stays_refused(poolmgr_run: PoolmgrRun) -> None:
    poolmgr_run.prod()
    assert refused_for(poolmgr_run, ALLOW, "10.0.0.0/8")


# ------------------------------------------------------------------ `loom-poolmgr run`


@pytest.mark.postgres
@pytest.mark.redis
def test_run_starts_outside_dev_with_the_flag_warns_and_exports_the_gauge(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`loom-poolmgr run` outside dev with the flag, `provider: local`, the CA and client
    certificate and no SSH key, bucket or bundle: the manager starts (its loop runs),
    `loom_poolmgr_mock_backend` is 1 on the metrics endpoint, a warning is logged, and a
    private `LOOM_POOLMGR_ALLOW_CIDRS` reaches the manager's allowlist."""
    h = poolmgr_run
    h.prod()
    h.mp.setenv(FLAG, "true")
    h.mp.setenv(ALLOW, "10.20.0.0/16")
    h.live(pg_url)
    h.serve_metrics()
    seen: dict[str, str] = {}

    async def scenario(manager: PoolManager, run: PoolmgrRun) -> None:
        await run.wait_tick(0)
        seen["metrics"] = await run.scrape()

    h.scenario = scenario
    with caplog.at_level(logging.WARNING):
        code = h.main("run")
    assert code in (0, None), f"loom-poolmgr run exited {code}"
    assert h.constructed, "no manager was built"
    assert "metrics" in seen, "the manager's loop never ran"
    assert sample(seen["metrics"], "loom_poolmgr_mock_backend") == 1.0, (
        "loom_poolmgr_mock_backend is not 1 on the metrics endpoint"
    )
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "mock" in r.getMessage().lower()
    ]
    assert warnings, "no warning about the mock backend at start"
    allow = {str(c) for c in h.constructed[0].get("extra_allow") or ()}
    assert allow == {"10.20.0.0/16"}, allow


@pytest.mark.parametrize("case", ["no_flag", "runpod", "public_allow_cidrs"])
def test_run_is_refused_outside_dev(
    poolmgr_run: PoolmgrRun, case: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 2 before any manager is built: `provider: local` without the flag, the flag
    with `provider: runpod`, or the flag with a public `LOOM_POOLMGR_ALLOW_CIDRS`; the
    refusals `production_problems` reports (a bad `ALLOW_CIDRS` may instead be refused
    when the settings load). The runpod setup is complete (its proxy bundle passes the
    sha check), so only the mock-backend rule can refuse it, and the refusal names it."""
    h = poolmgr_run
    h.prod()
    if case == "no_flag":
        pass
    elif case == "runpod":
        h.mp.setenv(FLAG, "true")
        everything_runpod_needs(h)
        any_bundle_matches_the_lock(h)
        h.mp.setenv("LOOM_POOLMGR_SERVING_YAML", str(serving_file(h.tmp / "s", provider="runpod")))
    else:
        h.mp.setenv(FLAG, "true")
        h.mp.setenv(ALLOW, "10.1.0.0/24,52.1.2.3")
    try:
        code = h.main("run")
    except ValueError as e:  # a bad ALLOW_CIDRS refused when the settings load
        assert case == "public_allow_cidrs" and ALLOW in str(e), e
        code = 2
    assert code == 2, f"loom-poolmgr run exited {code}"
    assert h.constructed == [], "a manager was built"
    if case == "runpod":
        out = capsys.readouterr()
        assert "mock" in (out.err + out.out).lower(), out
