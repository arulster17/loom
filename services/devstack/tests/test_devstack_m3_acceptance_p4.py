"""m3-p's reading of the deploy log (M3 spec §10 m3-p, §6.6 P4, §6.12): `p4_log_order`.

The log is JSON Lines, one record per step event: `{"ts", "release", "step", "status",
"service" (update only), "only" (apply start only), "detail"}` (§6.12). The order check:
from the release's last `apply`/`start` record, `update`/`gateway`/`ok` comes before
`update`/`poolmgr`/`start`. Records of other releases do not count; a line that is not a
JSON object is skipped. The refusal of `apply --only poolmgr` is the host's half
(`loom-deploy acceptance`), not the container's.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from loom_devstack import m3_acceptance as m3
from loom_devstack.acceptance import AcceptanceError, ItemResult

SHA = "abc1234def5678abc1234def5678abc1234def56"
OLD = "0123456789abcdef0123456789abcdef01234567"


def rec(step: str, status: str, svc: str | None = None, release: str = SHA, **kw: Any) -> str:
    r: dict[str, Any] = {"ts": "2026-11-18T10:00:00Z", "release": release, "step": step}
    r["status"] = status
    if svc:
        r["service"] = svc
    if step == "apply" and status == "start":
        r["only"] = kw.pop("only", None)
    r.update(kw)
    return json.dumps(r)


def update(svc: str, release: str = SHA, ok: bool = True) -> list[str]:
    return [
        rec("update", "start", svc, release),
        rec("update", "ok" if ok else "failed", svc, release),
    ]


def apply(
    release: str = SHA, services: tuple[str, ...] = ("billing", "gateway", "poolmgr", "web")
) -> list[str]:
    lines = [
        rec("apply", "start", release=release),
        rec("manifest", "start", release=release),
        rec("manifest", "ok", release=release),
        rec("render", "ok", release=release),
        rec("pull", "ok", release=release),
        rec("migrate", "start", release=release),
        rec("migrate", "ok", release=release),
    ]
    for s in services:
        lines += update(s, release)
    return [*lines, rec("record", "ok", release=release)]


def order(lines: list[str], release: str | None = SHA) -> tuple[int, int, int]:
    return m3.p4_log_order(lines, release)


def test_a_full_apply_in_order() -> None:
    lines = apply()
    start, gw_ok, pm_start = order(lines)
    assert start == 0
    assert json.loads(lines[gw_ok]) == json.loads(rec("update", "ok", "gateway"))
    assert json.loads(lines[pm_start]) == json.loads(rec("update", "start", "poolmgr"))


def test_poolmgr_started_before_the_gateway_is_refused() -> None:
    lines = [rec("apply", "start"), *update("poolmgr"), *update("gateway")]
    with pytest.raises(AcceptanceError, match="gateway"):
        order(lines)


def test_poolmgr_started_while_the_gateway_was_only_starting_is_refused() -> None:
    lines = [
        rec("apply", "start"),
        rec("update", "start", "gateway"),
        rec("update", "start", "poolmgr"),
        rec("update", "ok", "gateway"),
        rec("update", "ok", "poolmgr"),
    ]
    with pytest.raises(AcceptanceError, match="gateway"):
        order(lines)


def test_a_failed_gateway_update_is_not_ok() -> None:
    lines = [rec("apply", "start"), *update("gateway", ok=False), rec("update", "start", "poolmgr")]
    with pytest.raises(AcceptanceError, match="gateway"):
        order(lines)


def test_only_gateway_and_poolmgr() -> None:
    lines = [
        rec("apply", "start", only=["gateway", "poolmgr"]),
        *update("gateway"),
        *update("poolmgr"),
    ]
    assert order(lines) == (0, 2, 3)


def test_the_last_apply_of_the_release_is_judged() -> None:
    """A failed first attempt (gateway unhealthy), then a good retry: the retry counts."""
    first = [
        rec("apply", "start"),
        rec("update", "start", "gateway"),
        rec("update", "failed", "gateway"),
    ]
    lines = first + apply()
    start, _, _ = order(lines)
    assert start == len(first)


def test_a_good_earlier_apply_does_not_hide_a_bad_retry() -> None:
    lines = [*apply(), rec("apply", "start"), *update("poolmgr"), *update("gateway")]
    with pytest.raises(AcceptanceError, match="gateway"):
        order(lines)


def test_an_earlier_release_in_good_order_does_not_hide_this_one() -> None:
    lines = [*apply(OLD), rec("apply", "start"), *update("poolmgr"), *update("gateway")]
    with pytest.raises(AcceptanceError):
        order(lines)


def test_another_releases_gateway_ok_does_not_count() -> None:
    lines = [
        rec("apply", "start"),
        rec("update", "start", "gateway"),
        rec("update", "ok", "gateway", release=OLD),
        rec("update", "start", "poolmgr"),
    ]
    with pytest.raises(AcceptanceError, match="gateway"):
        order(lines)


def test_later_records_of_other_releases_are_ignored() -> None:
    """The host's P4 check logs a refused apply of another release after this one."""
    lines = [*apply(), rec("apply", "refused", release=OLD, detail="poolmgr ahead of the gateway")]
    assert order(lines)[0] == 0


def test_lines_that_are_not_json_objects_are_skipped() -> None:
    lines = ["", "not json", "[1, 2]", *apply()]
    assert order(lines)[0] == 3


def test_no_poolmgr_update_is_refused() -> None:
    lines = apply(services=("billing", "gateway", "web"))
    with pytest.raises(AcceptanceError, match="poolmgr"):
        order(lines)


def test_the_release_is_required() -> None:
    with pytest.raises(AcceptanceError, match="release"):
        order(apply(), None)
    with pytest.raises(AcceptanceError, match="release"):
        order(apply(), "")


def test_a_release_the_log_never_starts_is_refused() -> None:
    with pytest.raises(AcceptanceError, match=OLD[:7]):
        order(apply(), OLD)


def test_a_refused_apply_of_the_release_is_not_a_start() -> None:
    with pytest.raises(AcceptanceError, match=SHA[:7]):
        order([rec("apply", "refused", detail="tmpfs")])


def acc_for(tmp_path: Path, release: str | None, lines: list[str]) -> m3.M3Acceptance:
    log = tmp_path / "deploy.log"
    log.write_text("\n".join(lines) + "\n")
    args = SimpleNamespace(release=release, deploy_log=str(log))
    return cast(m3.M3Acceptance, SimpleNamespace(args=args))


def test_m3_p_refuses_to_run_without_a_release(tmp_path: Path) -> None:
    acc = acc_for(tmp_path, None, apply())
    with pytest.raises(AcceptanceError, match="--release"):
        m3.M3Acceptance.item_m3_p(acc, ItemResult("m3-p", "P4"))


def test_m3_p_reads_only_the_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The container half runs no deploy command and no docker (the refusal moved to the
    host, §6.12)."""

    def no_subprocess(*a: Any, **k: Any) -> Any:
        raise AssertionError(f"m3-p ran a command: {a}")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    monkeypatch.setattr(subprocess, "Popen", no_subprocess)
    res = ItemResult("m3-p", "P4")
    m3.M3Acceptance.item_m3_p(acc_for(tmp_path, SHA, apply()), res)
    assert res.status == "PASS"
    assert res.numbers["deploy_log_lines"]


def test_m3_p_fails_on_a_bad_order(tmp_path: Path) -> None:
    lines = [rec("apply", "start"), *update("poolmgr"), *update("gateway")]
    with pytest.raises(AcceptanceError):
        m3.M3Acceptance.item_m3_p(acc_for(tmp_path, SHA, lines), ItemResult("m3-p", "P4"))
