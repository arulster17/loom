"""The apply-in-progress marker (FU3; M3 spec §6.6 "An apply that did not finish",
Rollback; §6.12 "The apply-in-progress marker", Deploy log; §13 A12).

`loom-deploy` runs as on the box against the fakes of `devstack_m3_kit.py` (Linux only, like
the other deploy tests: the deploy root is under /dev/shm). An update is made to fail by
setting the fake docker's health of a service to `starting` with a 2 s health timeout.

What these tests take from the spec:
- every apply (full, `--only`, rollback) writes `<root>/var/lib/loom/apply-in-progress.json`
  after migrate's exit 0 and before its first `up`, and removes it in the `record` step;
- a failed apply leaves it: `0644`, JSON with exactly `release`, `only`, `started_at`;
- while anything exists at that path, every `apply --only` is refused (exit 2, one
  `apply`/`refused` record, before any fetch), naming the marker's path and the full apply to
  run, never the gateway, never the marker's `only`; `acceptance` is refused too (exit 2,
  nothing run, nothing logged); `status` prints the marker on its first line;
- a full apply or rollback runs as usual with the marker set, updates Redis too, and
  removes the marker when it records;
- a marker that cannot be written fails the apply (exit 1, only `apply`/`failed`) before any
  `up`; writing and removing the marker log no record of their own.

"Message" below is the refused record's `detail` plus stderr.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from devstack_m3_kit import (
    ORDER,
    UP_VALUED,
    Box,
    aws_op,
    compose_calls,
    deploy_root_base,
    mode,
    non_options,
    runs,
    secret_values,
    site,
    ups,
)

SHA_A = "1a" * 20
SHA_B = "2b" * 20
SHA_C = "3c" * 20
STEPS = ("apply", "manifest", "render", "pull", "migrate", "update", "record")
STATUSES = ("start", "ok", "failed", "refused")
FAST = {"LOOM_DEPLOY_HEALTH_TIMEOUT_S": "2"}
TS = "2026-11-20T00:00:00+00:00"


def assert_exit(proc: Any, code: int) -> None:
    """`loom-deploy` ran (an import error also exits 1) and exited with `code`."""
    assert "No module named" not in proc.stderr, ran(proc)
    assert proc.returncode == code, ran(proc)


def ran(proc: Any) -> str:
    out, err = proc.stdout[-3000:], proc.stderr[-3000:]
    return f"exit {proc.returncode}\n--- stdout\n{out}\n--- stderr\n{err}"


@pytest.fixture
def box() -> Iterator[Box]:
    b = Box(deploy_root_base(), site_doc=site())
    yield b
    b.close()


def marker(box: Box) -> Path:
    return box.var_lib / "apply-in-progress.json"


def set_health(box: Box, health: dict[str, str]) -> None:
    box.docker_cfg = box.load_docker()  # the fake docker rewrites docker.json
    box.docker_cfg["health"] = health
    box.save_docker()


def apply(box: Box, sha: str, *only: str, **env: str) -> Any:
    args = ["apply", "--release", sha]
    for svc in only:
        args += ["--only", svc]
    return box.deploy(*args, **env)


def applied(box: Box, sha: str) -> None:
    """Publish `sha` (if not yet) and apply it in full, successfully."""
    if sha not in box.releases:
        box.publish(sha)
    proc = apply(box, sha)
    assert proc.returncode == 0, f"applying {sha[:4]} first: {ran(proc)}"


def fail_full_apply(box: Box, sha: str, at: str = "gateway") -> Any:
    """A full apply of `sha` whose update of `at` times out (exit 1); health cleared after."""
    if sha not in box.releases:
        box.publish(sha)
    set_health(box, {at: "starting"})
    proc = apply(box, sha, **FAST)
    set_health(box, {})
    assert_exit(proc, 1)
    return proc


def failed_b_over_a(box: Box) -> bytes:
    """A applied; a full apply of B fails at the gateway. Returns release.json from before B."""
    applied(box, SHA_A)
    before = (box.var_lib / "release.json").read_bytes()
    fail_full_apply(box, SHA_B)
    return before


def write_marker(box: Box, content: str | dict[str, Any], perms: int = 0o644) -> Path:
    p = marker(box)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content if isinstance(content, str) else json.dumps(content))
    p.chmod(perms)
    return p


def read_marker(box: Box) -> dict[str, Any]:
    p = marker(box)
    assert os.path.lexists(p), f"no marker at {p} after an apply that did not finish"
    doc = json.loads(p.read_text())
    assert isinstance(doc, dict), doc
    return doc


def assert_no_marker(box: Box) -> None:
    p = marker(box)
    assert not os.path.lexists(p), f"the marker is still there: {p}"


def message(box: Box, proc: Any, n_log: int) -> str:
    refused = [r for r in box.log_records()[n_log:] if r.get("status") == "refused"]
    detail = str(refused[-1].get("detail") or "") if refused else ""
    return detail + "\n" + proc.stderr


def fetched(tl: list[dict[str, Any]]) -> list[list[str]]:
    """The `aws s3`/`aws s3api` calls (the release's fetch) in `tl`."""
    s3 = (["s3"], ["s3api"])
    return [r["argv"] for r in tl if r["tool"] == "aws" and aws_op(r["argv"])[:1] in s3]


def assert_refused_by_marker(box: Box, proc: Any, n_log: int, n_tl: int) -> str:
    """§6.12: exit 2, one `apply`/`refused` record (and nothing else), naming the marker's
    path and never the gateway; nothing fetched, no compose pull/run/up. Returns the message."""
    assert_exit(proc, 2)
    new = box.log_records()[n_log:]
    assert [(r["step"], r["status"]) for r in new] == [("apply", "refused")], new
    said = message(box, proc, n_log)
    assert str(marker(box)) in said, f"the refusal does not name {marker(box)}: {said}"
    assert "gateway" not in said.lower(), f"the marker's refusal names the gateway: {said}"
    tl = box.timeline()[n_tl:]
    assert not fetched(tl), f"a refused apply fetched: {fetched(tl)}"
    subs = [sub for _, _, sub, _ in compose_calls(tl)]
    assert not set(subs) & {"pull", "run", "up"}, f"a refused apply ran compose {subs}"
    return said


def tree(root: Path) -> dict[str, tuple[str, bytes, int, int]]:
    out = {}
    for p in root.rglob("*"):
        st = p.lstat()
        rel = str(p.relative_to(root))
        if stat.S_ISREG(st.st_mode):
            out[rel] = ("file", p.read_bytes(), stat.S_IMODE(st.st_mode), st.st_mtime_ns)
        else:
            out[rel] = ("other", b"", stat.S_IMODE(st.st_mode), 0)
    return out


def up_services(box: Box, n_tl: int) -> list[str]:
    return [s for _, s in ups(box.timeline()[n_tl:])]


# ---------------------------------------------------------------- written, removed


def test_every_up_sees_the_marker_and_success_removes_it(box: Box) -> None:
    """§6.6 step 4: the marker is written before the first `up`; step 5 removes it."""
    wrap = box.base / "wrap"
    wrap.mkdir()
    seen = box.base / "seen.jsonl"
    fake = box.bin / "docker"
    script = wrap / "docker"
    script.write_text(
        f"#!{sys.executable} -IS\n"
        "import json, os, sys\n"
        "argv = sys.argv[1:]\n"
        "if argv[:1] == ['compose']:\n"
        f"    with open({str(seen)!r}, 'a') as f:\n"
        "        f.write(json.dumps({'argv': argv, 'marker': "
        f"os.path.lexists({str(marker(box))!r})}}) + '\\n')\n"
        f"os.execv({str(fake)!r}, [{str(fake)!r}, *argv])\n"
    )
    script.chmod(0o755)
    box.publish(SHA_A)
    proc = apply(box, SHA_A, PATH=f"{wrap}:{box.bin}:/usr/local/bin:/usr/bin:/bin")
    assert_exit(proc, 0)
    lines = [json.loads(x) for x in seen.read_text().splitlines() if x.strip()]
    records = [{"tool": "docker", "argv": x["argv"]} for x in lines]
    at_up = [lines[n]["marker"] for n, _, sub, _ in compose_calls(records) if sub == "up"]
    assert len(at_up) == len(ORDER), f"{len(at_up)} compose ups seen, not {len(ORDER)}"
    assert all(at_up), f"marker present at each up: {at_up}"
    assert_no_marker(box)


def test_a_failed_full_apply_leaves_the_marker(box: Box) -> None:
    before = failed_b_over_a(box)
    p = marker(box)
    assert os.path.lexists(p), f"a failed apply left no marker at {p}"
    assert stat.S_ISREG(p.lstat().st_mode), "the marker is not a regular file"
    assert mode(p) == 0o644, f"marker mode {oct(mode(p))}, not 0o644"
    text = p.read_text()
    doc = json.loads(text)
    assert isinstance(doc, dict) and set(doc) == {"release", "only", "started_at"}, doc
    assert doc["release"] == SHA_B and doc["only"] is None, doc
    started = dt.datetime.fromisoformat(str(doc["started_at"]).replace("Z", "+00:00"))
    assert started.utcoffset() == dt.timedelta(0), f"started_at not UTC: {doc['started_at']}"
    for v in [*secret_values(box.store), box.ecr_token]:
        assert v not in text, "the marker holds a secret value"
    assert (box.var_lib / "release.json").read_bytes() == before, "release.json changed"


# ---------------------------------------------------------------- --only refused


def test_only_is_refused_after_a_failed_apply(box: Box) -> None:
    failed_b_over_a(box)
    before = tree(box.root)
    n_log, n_tl = len(box.log_records()), len(box.timeline())
    proc = apply(box, SHA_A, "poolmgr")
    said = assert_refused_by_marker(box, proc, n_log, n_tl)
    assert "apply-in-progress.json" in said, said
    assert f"loom-deploy apply --release {SHA_A}" in said, f"no full apply of A: {said}"
    assert f"--release {SHA_B}" in said, f"no full apply of B: {said}"
    after = tree(box.root)
    log_key = "var/lib/loom/deploy.log"
    assert after[log_key][1].startswith(before[log_key][1]), "deploy.log was not appended to"
    before.pop(log_key)
    after.pop(log_key)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert not changed, f"a refused apply changed files under the root: {changed}"


@pytest.mark.parametrize(
    "only, sha",
    [
        pytest.param(["gateway", "poolmgr"], SHA_B, id="gateway-poolmgr-B"),
        pytest.param(["web"], SHA_A, id="web-A"),
        pytest.param(["redis"], SHA_B, id="redis-B"),
        pytest.param(["poolmgr"], SHA_B, id="poolmgr-B"),
    ],
)
def test_every_only_is_refused_while_the_marker_is_set(box: Box, only: list[str], sha: str) -> None:
    failed_b_over_a(box)
    n_log, n_tl = len(box.log_records()), len(box.timeline())
    proc = apply(box, sha, *only)
    assert_refused_by_marker(box, proc, n_log, n_tl)
    assert not up_services(box, n_tl)


def test_a_failed_only_apply_stores_only_as_given(box: Box) -> None:
    applied(box, SHA_A)
    box.publish(SHA_B)
    n_log = len(box.log_records())
    set_health(box, {"gateway": "starting"})
    proc = apply(box, SHA_B, "poolmgr,gateway", **FAST)
    set_health(box, {})
    assert_exit(proc, 1)
    doc = read_marker(box)
    assert doc.get("only") == ["poolmgr", "gateway"], doc
    new = box.log_records()[n_log:]
    starts = [r for r in new if (r["step"], r["status"]) == ("apply", "start")]
    assert len(starts) == 1 and starts[0].get("only") == doc["only"], starts
    n_log, n_tl = len(box.log_records()), len(box.timeline())
    refused = apply(box, SHA_A, "poolmgr")
    assert_refused_by_marker(box, refused, n_log, n_tl)


UNREADABLE = [
    pytest.param("", id="empty"),
    pytest.param("not json", id="not-json"),
    pytest.param("[]", id="list"),
    pytest.param("{}", id="empty-object"),
    pytest.param({"release": "gateway"}, id="release-not-a-sha"),
    pytest.param({"release": None}, id="release-null"),
    pytest.param({"release": SHA_B, "only": "gateway", "started_at": TS}, id="only-a-string"),
    pytest.param({"release": SHA_B, "only": ["nope"], "started_at": TS}, id="only-unknown"),
    pytest.param({"release": SHA_B, "only": None, "started_at": 5}, id="started-at-a-number"),
    pytest.param({"release": SHA_B, "only": ["gateway"], "started_at": TS}, id="valid-only"),
    pytest.param("mode-000", id="valid-mode-000"),
]


@pytest.mark.parametrize("content", UNREADABLE)
def test_a_marker_that_cannot_be_read_still_refuses(box: Box, content: Any) -> None:
    """Whatever the marker holds, `--only` is refused; the message never prints `only`
    (no "gateway") and has no "None"/"null"; a full apply then clears it."""
    applied(box, SHA_A)
    if content == "mode-000":
        write_marker(box, {"release": SHA_B, "only": None, "started_at": TS}, 0o000)
    else:
        write_marker(box, content)
    n_log, n_tl = len(box.log_records()), len(box.timeline())
    proc = apply(box, SHA_A, "poolmgr")
    said = assert_refused_by_marker(box, proc, n_log, n_tl)
    assert "None" not in said and "null" not in said, said
    again = apply(box, SHA_A)
    assert_exit(again, 0)
    assert_no_marker(box)


def test_a_first_ever_failed_apply_refuses_only_naming_its_release(box: Box) -> None:
    fail_full_apply(box, SHA_B)
    assert read_marker(box).get("release") == SHA_B
    assert not (box.var_lib / "release.json").exists(), "a failed first apply recorded"
    n_log, n_tl = len(box.log_records()), len(box.timeline())
    proc = apply(box, SHA_B, "gateway", "poolmgr")
    said = assert_refused_by_marker(box, proc, n_log, n_tl)
    assert f"loom-deploy apply --release {SHA_B}" in said, said
    assert "None" not in said and "null" not in said, said


def test_a_stop_after_record_is_cleared_by_a_full_apply(box: Box) -> None:
    """Stopped after `release.json` was written, before the removal: `--only` stays
    refused, and a full apply of the same release clears it (and updates Redis too)."""
    applied(box, SHA_A)
    applied(box, SHA_B)
    write_marker(box, {"release": SHA_B, "only": None, "started_at": TS})
    n_log, n_tl = len(box.log_records()), len(box.timeline())
    proc = apply(box, SHA_B, "poolmgr")
    said = assert_refused_by_marker(box, proc, n_log, n_tl)
    assert SHA_B in said, said
    n_tl = len(box.timeline())
    full = apply(box, SHA_B)
    assert_exit(full, 0)
    assert up_services(box, n_tl) == ORDER
    assert_no_marker(box)


# ---------------------------------------------------------------- full apply, rollback


def test_a_full_apply_after_a_failed_one_also_updates_redis(box: Box) -> None:
    failed_b_over_a(box)
    n_tl = len(box.timeline())
    proc = apply(box, SHA_B)
    assert_exit(proc, 0)
    assert up_services(box, n_tl) == ORDER
    redis_ups = [
        rest
        for _, _, sub, rest in compose_calls(box.timeline()[n_tl:])
        if sub == "up" and "redis" in non_options(rest, set(UP_VALUED))
    ]
    assert len(redis_ups) == 1, redis_ups
    assert "--force-recreate" not in redis_ups[0], f"redis was recreated: {redis_ups[0]}"
    assert_no_marker(box)
    rec = json.loads((box.var_lib / "release.json").read_text())
    assert rec["release"] == SHA_B, rec


def test_rollback_after_a_failed_apply_goes_to_previous(box: Box) -> None:
    applied(box, SHA_A)
    applied(box, SHA_B)
    fail_full_apply(box, SHA_C)
    n_tl = len(box.timeline())
    proc = box.deploy("rollback")
    assert_exit(proc, 0)
    rec = json.loads((box.var_lib / "release.json").read_text())
    assert (rec["release"], rec["previous"]) == (SHA_A, SHA_B), rec
    assert up_services(box, n_tl) == ORDER
    assert_no_marker(box)


def test_a_failed_rollback_keeps_the_marker(box: Box) -> None:
    applied(box, SHA_A)
    applied(box, SHA_B)
    set_health(box, {"gateway": "starting"})
    proc = box.deploy("rollback", **FAST)
    set_health(box, {})
    assert_exit(proc, 1)
    doc = read_marker(box)
    assert set(doc) == {"release", "only", "started_at"}, doc
    assert doc["release"] == SHA_A and doc["only"] is None, doc
    assert isinstance(doc["started_at"], str), doc
    n_log, n_tl = len(box.log_records()), len(box.timeline())
    refused = apply(box, SHA_B, "poolmgr")
    assert_refused_by_marker(box, refused, n_log, n_tl)
    full = apply(box, SHA_B)
    assert_exit(full, 0)
    assert_no_marker(box)
    only = apply(box, SHA_B, "poolmgr")
    assert_exit(only, 0)


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(SHA_A, id="the-recorded-release"),
        pytest.param(SHA_B, id="the-markers-release"),
        pytest.param(SHA_C, id="a-third-release"),
    ],
)
def test_any_successful_full_apply_clears_the_marker(box: Box, target: str) -> None:
    applied(box, SHA_A)
    box.publish(SHA_B)
    box.publish(SHA_C)
    write_marker(box, {"release": SHA_B, "only": None, "started_at": TS})
    n_tl = len(box.timeline())
    proc = apply(box, target)
    assert_exit(proc, 0)
    assert_no_marker(box)
    assert up_services(box, n_tl) == ORDER, "a full apply with the marker set updates redis too"


# ---------------------------------------------------------------- status, acceptance


def test_status_shows_the_marker(box: Box) -> None:
    failed_b_over_a(box)
    proc = box.deploy("status")
    assert_exit(proc, 0)
    lines = proc.stdout.splitlines()
    first = lines[0] if lines else ""
    assert str(marker(box)) in first, f"status's first line: {first!r}"
    assert f"release {SHA_B}" in first, first
    assert f"recorded: {SHA_A}" in first, first
    assert "refused until a full apply" in first, first


def test_acceptance_is_refused_while_the_marker_is_set(box: Box) -> None:
    applied(box, SHA_A)
    box.publish(SHA_B)
    first = box.deploy("acceptance", "--p4-release", SHA_B)
    assert_exit(first, 0)
    acc = box.var_lib / "acceptance"
    host = acc / "host.json"
    assert host.is_file(), "the first acceptance run wrote no host.json"
    report = acc / "report.json"
    report.write_text('{"items": [], "note": "an older run"}\n')
    fail_full_apply(box, SHA_B)
    old_host, old_report = host.read_bytes(), report.read_bytes()
    n_log, n_tl = len(box.log_records()), len(box.timeline())
    proc = box.deploy("acceptance", "--p4-release", SHA_B)
    assert_exit(proc, 2)
    assert str(marker(box)) in proc.stderr, ran(proc)
    assert "loom-deploy apply --release" in proc.stderr, ran(proc)
    assert not [r for _, svc, r in runs(box.timeline()[n_tl:]) if svc == "acceptance"]
    assert box.log_records()[n_log:] == [], "a refused acceptance logged a record"
    assert host.read_bytes() == old_host and report.read_bytes() == old_report


def test_a_first_ever_failed_apply_status_and_acceptance(box: Box) -> None:
    fail_full_apply(box, SHA_B)
    proc = box.deploy("status")
    assert_exit(proc, 0)
    lines = proc.stdout.splitlines()
    first = lines[0] if lines else ""
    assert str(marker(box)) in first, f"status's first line: {first!r}"
    assert f"release {SHA_B}" in first and "recorded: none" in first, first
    assert "no release recorded" in proc.stdout, proc.stdout
    box.publish(SHA_A)
    acc = box.deploy("acceptance", "--p4-release", SHA_A)
    assert_exit(acc, 2)
    assert str(marker(box)) in acc.stderr, ran(acc)


# ---------------------------------------------------------------- edges


def test_a_marker_that_cannot_be_written_fails_cleanly(box: Box) -> None:
    applied(box, SHA_A)
    p = marker(box)
    p.mkdir()
    (p / "keep").write_text("x")
    n_log, n_tl = len(box.log_records()), len(box.timeline())
    proc = apply(box, SHA_A)
    assert_exit(proc, 1)
    assert not up_services(box, n_tl), "a service was updated without a marker"
    new = box.log_records()[n_log:]
    failed = [r for r in new if r["status"] == "failed"]
    assert [(r["step"], r["status"]) for r in failed] == [("apply", "failed")], new
    assert "apply-in-progress.json" in str(failed[0].get("detail")), failed[0]
    assert any((r["step"], r["status"]) == ("migrate", "ok") for r in new), new
    assert p.is_dir() and (p / "keep").read_text() == "x", "the marker already there changed"


def test_a_failed_migration_writes_no_marker(box: Box) -> None:
    box.publish(SHA_A)
    box.docker_cfg = box.load_docker()
    box.docker_cfg["run_exit"] = {"migrate": 3}
    box.save_docker()
    proc = apply(box, SHA_A)
    assert_exit(proc, 1)
    assert not os.path.lexists(marker(box)), "an apply that failed at migrate wrote a marker"


@pytest.mark.parametrize("case, code", [("site-lacks-domain", 2), ("compose-sha256-wrong", 1)])
def test_a_refused_or_early_failed_full_apply_leaves_the_marker(
    box: Box, case: str, code: int
) -> None:
    applied(box, SHA_A)
    p = write_marker(box, {"release": SHA_B, "only": None, "started_at": TS})
    before = p.read_bytes()
    if case == "site-lacks-domain":
        box.set_site(site(domain=None))
        proc = apply(box, SHA_A)
    else:
        wrong = {"path": f"releases/{SHA_B}/compose.prod.yml", "sha256": "0" * 64}
        box.publish(SHA_B, manifest_over={"compose": wrong})
        proc = apply(box, SHA_B)
    assert_exit(proc, code)
    assert p.is_file() and p.read_bytes() == before, "the marker already there changed"


def test_a_successful_only_apply_leaves_no_marker_and_p4_holds(box: Box) -> None:
    applied(box, SHA_A)
    box.publish(SHA_B)
    proc = apply(box, SHA_B, "gateway", "poolmgr")
    assert_exit(proc, 0)
    assert_no_marker(box)
    n_log = len(box.log_records())
    refused = apply(box, SHA_A, "poolmgr")
    assert_exit(refused, 2)
    assert "gateway" in message(box, refused, n_log).lower(), ran(refused)


def test_the_marker_adds_no_deploy_log_values(box: Box) -> None:
    failed_b_over_a(box)
    proc = apply(box, SHA_B)
    assert_exit(proc, 0)
    recs = box.log_records()
    assert recs
    for r in recs:
        assert r["step"] in STEPS and r["status"] in STATUSES, r
        if r["step"] == "update":
            assert r.get("service") in ORDER, r
