"""m3-p's reading of the deploy log (M3 spec §10 m3-p, §6.6 P4): `p4_log_order`.

The log's format is not fixed by the spec, so the reader accepts plain lines, timings
(`waited 4.1 s`), `docker compose` progress lines and JSON-ish records; it reads the
release's last apply (not a closing `done` line), and refuses a log where the pool manager
was updated before a record that the gateway was healthy. The release is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from loom_devstack import m3_acceptance as m3
from loom_devstack.acceptance import AcceptanceError, ItemResult

SHA = "abc1234def5678abc1234def5678abc1234def56"
OLD = "0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0"


def order(text: str, release: str | None = SHA) -> tuple[int, int, int]:
    return m3.p4_log_order(text.strip("\n").splitlines(), release)


PLAIN = f"""
2026-11-18T10:00:00Z apply {SHA}: start
2026-11-18T10:00:01Z manifest ok
2026-11-18T10:00:09Z migrate: exit 0
2026-11-18T10:00:10Z billing: up -d --no-deps
2026-11-18T10:00:14Z billing healthy
2026-11-18T10:00:15Z gateway: up -d --no-deps
2026-11-18T10:00:19Z gateway healthy
2026-11-18T10:00:20Z poolmgr: up -d --no-deps
2026-11-18T10:00:24Z poolmgr healthy
"""


def test_plain_lines() -> None:
    assert order(PLAIN) == (5, 7, 6)


@pytest.mark.parametrize(
    "healthy",
    [
        "gateway healthy (waited 4.1 s)",
        "waited 4.1 s for gateway: healthy",
        "gateway: healthy after 4.1 s",
        "Container loom-gateway-1  Healthy",
        "gateway is healthy",
    ],
)
def test_a_healthy_record_with_timings_counts(healthy: str) -> None:
    text = f"apply {SHA}\ngateway: updating\n{healthy}\npoolmgr: updating\n"
    assert order(text) == (1, 3, 2)


UPDATE_THEN_WAIT = [
    "{svc}: up -d --no-deps {svc}, then wait healthy",  # §6.6 step 4's wording
    "{svc}: up -d, waiting until healthy",
    "updating {svc} (wait for healthy)",
]


@pytest.mark.parametrize("update", UPDATE_THEN_WAIT)
def test_an_update_line_that_names_its_pending_wait_is_an_update(update: str) -> None:
    gw, pm = update.format(svc="gateway"), update.format(svc="poolmgr")
    text = f"apply {SHA}: start\n{gw}\ngateway healthy\n{pm}\npoolmgr healthy\n"
    assert order(text) == (1, 3, 2)


@pytest.mark.parametrize("update", UPDATE_THEN_WAIT)
def test_an_update_line_that_names_its_pending_wait_in_the_wrong_order_is_refused(
    update: str,
) -> None:
    gw, pm = update.format(svc="gateway"), update.format(svc="poolmgr")
    with pytest.raises(AcceptanceError, match="gateway"):
        order(f"apply {SHA}: start\n{pm}\n{gw}\ngateway healthy\n")
    with pytest.raises(AcceptanceError, match="healthy"):
        order(f"apply {SHA}: start\n{gw}\n{pm}\ngateway healthy\n")


def test_a_closing_apply_done_line_does_not_start_the_apply() -> None:
    text = PLAIN + f"2026-11-18T10:01:00Z apply {SHA}: done, recorded as current\n"
    assert order(text) == (5, 7, 6)


def test_the_last_apply_of_the_release_is_read() -> None:
    """A failed first attempt, then a retry: the retry is the one judged."""
    text = (
        f"apply {SHA}: start\n"
        "gateway: up\n"
        "gateway unhealthy after 120 s\n"
        f"apply {SHA}: failed\n"
        f"apply {SHA}: start\n"
        "gateway: up\n"
        "gateway healthy\n"
        "poolmgr: up\n"
        f"apply {SHA}: done\n"
    )
    assert order(text) == (5, 7, 6)


def test_an_earlier_release_in_good_order_does_not_hide_this_one() -> None:
    text = (
        f"apply {OLD}: start\ngateway: up\ngateway healthy\npoolmgr: up\napply {OLD}: done\n"
        f"apply {SHA}: start\npoolmgr: up\ngateway: up\ngateway healthy\n"
    )
    with pytest.raises(AcceptanceError):
        order(text)


def test_json_lines() -> None:
    records = [
        {"ts": "2026-11-18T10:00:00Z", "event": "apply_start", "release": SHA},
        {"event": "service_update", "service": "billing", "release": SHA},
        {"event": "wait_healthy", "service": "billing", "result": "healthy", "waited_s": 3.0},
        {"event": "service_update", "service": "gateway", "release": SHA},
        {"event": "wait_healthy", "service": "gateway", "result": "healthy", "waited_s": 4.1},
        {"event": "service_update", "service": "poolmgr", "release": SHA},
        {"event": "apply_done", "release": SHA},
    ]
    text = "\n".join(json.dumps(r) for r in records)
    assert order(text) == (3, 5, 4)


def test_json_lines_with_a_pending_wait_only_are_refused() -> None:
    records = [
        {"event": "apply_start", "release": SHA},
        {"event": "service_update", "service": "gateway"},
        {"event": "wait_healthy", "service": "gateway"},
        {"event": "service_update", "service": "poolmgr"},
    ]
    with pytest.raises(AcceptanceError, match="healthy"):
        order("\n".join(json.dumps(r) for r in records))


def test_poolmgr_before_gateway_is_refused() -> None:
    text = f"apply {SHA}: start\npoolmgr: up\ngateway: up\ngateway healthy\n"
    with pytest.raises(AcceptanceError, match="gateway"):
        order(text)


def test_poolmgr_before_the_gateway_is_healthy_is_refused() -> None:
    text = f"apply {SHA}: start\ngateway: up\npoolmgr: up\ngateway healthy\n"
    with pytest.raises(AcceptanceError, match="healthy"):
        order(text)


@pytest.mark.parametrize(
    "not_healthy",
    [
        "gateway unhealthy",
        "gateway is not healthy",
        "gateway not yet healthy",
        "waiting for gateway to be healthy",
        "gateway: waiting until healthy",
        "gateway healthy: false",
        "gateway failed to become healthy",
        "timed out waiting for gateway healthy",
    ],
)
def test_a_gateway_that_is_not_healthy_is_refused(not_healthy: str) -> None:
    text = f"apply {SHA}: start\ngateway: up\n{not_healthy}\npoolmgr: up\n"
    with pytest.raises(AcceptanceError, match="healthy"):
        order(text)


def test_a_plan_line_is_not_a_step() -> None:
    text = (
        f"apply {SHA}: start\n"
        "plan: billing, gateway, poolmgr, web, caddy\n"
        "update order: gateway then poolmgr\n"
        "gateway: up\ngateway healthy\npoolmgr: up\n"
    )
    assert order(text) == (3, 5, 4)


def test_the_release_is_required() -> None:
    with pytest.raises(AcceptanceError, match="release"):
        order(PLAIN, None)
    with pytest.raises(AcceptanceError, match="release"):
        order(PLAIN, "")


def test_a_release_the_log_never_names_is_refused() -> None:
    with pytest.raises(AcceptanceError, match=OLD[:7]):
        order(PLAIN, OLD)


def test_m3_p_refuses_to_run_without_a_release(tmp_path: Path) -> None:
    log = tmp_path / "deploy.log"
    log.write_text(PLAIN)
    acc = SimpleNamespace(args=SimpleNamespace(release=None, deploy_log=str(log), p4_release=OLD))
    with pytest.raises(AcceptanceError, match="--release"):
        m3.M3Acceptance.item_m3_p(cast(m3.M3Acceptance, acc), ItemResult("m3-p", "P4"))


def test_the_default_deploy_log_is_documented_as_an_assumption() -> None:
    action = next(a for a in m3.parser()._actions if "--deploy-log" in a.option_strings)
    assert action.default
    assert "assumption" in (action.help or "")
