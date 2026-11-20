"""`loom-deploy`: edge cases of P4, the site parameter, D10, env files and re-rendering
(M3 spec §6.3, §6.6, §6.11, §6.12, §9; A12, A13, A29).

The in-process checks call the module's functions (`plan_steps`, `check_database_url` and
`Refused` are §6.12's interfaces; `validate_site`, `quote_env`/`env_text` and
`check_p4_refusal` are the functions behind §6.11's refusal, §6.12's env files and m3-p's
host half). The apply checks run `loom-deploy` with the kit's fake `aws`, `docker` and
`chown` (`devstack_m3_kit.Box`).

Env files are read back as compose reads an `env_file` (compose-go's dotenv): a
single-quoted value is literal; a double-quoted one ends at the first `"` not preceded by
a backslash, has its backslash escapes expanded (Go rune escapes, `\\$` meaning `$$`), then
its `$` interpolated (`$$` is a literal `$`).
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from devstack_m3_kit import (
    ORDER,
    Box,
    aws_op,
    deploy_root_base,
    ecr_images,
    full_store,
    site,
)

SHA_A = "a1" * 20
SHA_B = "b2" * 20
LO = "1" * 40  # sorts below HI
HI = "e" * 40
REDIS_LO = "docker.io/library/redis@sha256:" + "1" * 64
REDIS_HI = "docker.io/library/redis@sha256:" + "f" * 64
RDS_QUERY = "?sslmode=verify-full&sslrootcert=/etc/loom/rds-ca.pem"


def deploy() -> Any:
    return importlib.import_module("loom_devstack.deploy")


def ran(proc: Any) -> str:
    return (
        f"exit {proc.returncode}\n--- stdout\n{proc.stdout[-3000:]}\n"
        f"--- stderr\n{proc.stderr[-3000:]}"
    )


def assert_exit(proc: Any, code: int) -> None:
    assert "No module named" not in proc.stderr, ran(proc)
    assert proc.returncode == code, ran(proc)


def manifest(sha: str, redis: str | None = None) -> dict[str, Any]:
    images = dict(ecr_images(sha))
    if redis is not None:
        images["redis"] = redis
    return {
        "sha": sha,
        "images": images,
        "migrations": 12,
        "compose": {"path": f"releases/{sha}/compose.prod.yml", "sha256": "0" * 64},
        "bundle": {"path": f"releases/{sha}/deploy.tar.gz", "sha256": "1" * 64},
    }


def plan_ups(steps: Any) -> list[str]:
    return [svc for _, svc in ups_of(steps)]


def ups_of(steps: Any) -> list[tuple[str, str]]:
    out = []
    for s in steps:
        assert isinstance(s, tuple) and s, f"step {s!r} is not (action, *args)"
        if s[0] == "up":
            out.append(("up", s[1]))
    return out


def refusal(call: Any) -> str:
    """The call raises `deploy.Refused` naming the gateway; returns its message."""
    d = deploy()
    with pytest.raises(Exception) as caught:
        call()
    err = caught.value
    assert isinstance(err, d.Refused), f"not a deploy.Refused: {err!r}"
    assert "gateway" in str(err).lower(), f"the refusal does not name the gateway: {err}"
    return str(err)


def files_under(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def s3_calls(box: Box) -> list[list[str]]:
    return [a for a in box.calls("aws") if aws_op(a)[:1] in (["s3"], ["s3api"])]


# ---------------------------------------------------------------- reading env files as compose


_GO_ESCAPES = {
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
}


def _unescape(match: re.Match[str]) -> str:
    text = match.group(0)
    if text == "\\$":
        return "$$"
    return _GO_ESCAPES.get(text[1], text)


def _interpolate(text: str, raw: str) -> str:
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c == "$":
            assert text[i + 1 : i + 2] == "$", f"compose would interpolate `$` in {raw!r}"
            out.append("$")
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def compose_value(raw: str) -> str:
    """One env-file value (the text after `KEY=`) as compose reads it."""
    if raw.startswith("'"):
        end = raw.find("'", 1)
        assert end == len(raw) - 1, f"text after the closing quote: {raw!r}"
        return raw[1:end]
    if raw.startswith('"'):
        end = next((i for i in range(1, len(raw)) if raw[i] == '"' and raw[i - 1] != "\\"), -1)
        assert end == len(raw) - 1, f"the double-quoted value does not end at the end: {raw!r}"
        return _interpolate(re.sub(r"\\.", _unescape, raw[1:end]), raw)
    return _interpolate(raw.strip(), raw)


def compose_env_file(text: str) -> dict[str, str]:
    out = {}
    for line in text.split("\n"):
        if not line.strip():
            continue
        key, sep, value = line.partition("=")
        assert sep, f"not a KEY=value line: {line!r}"
        out[key] = compose_value(value)
    return out


# ---------------------------------------------------------------- 1. a refused --only apply


@pytest.fixture
def box() -> Iterator[Box]:
    b = Box(deploy_root_base(), site_doc=site())
    yield b
    b.close()


def test_a_first_only_poolmgr_apply_is_refused_before_any_fetch(box: Box) -> None:
    """A12, §6.12: with no release recorded, `apply --only poolmgr` would run a poolmgr
    while the gateway runs nothing (P4): refused before the release is fetched, nothing
    written, and the deploy log holds exactly one `apply`/`refused` record."""
    box.publish(SHA_B)
    proc = box.deploy("apply", "--release", SHA_B, "--only", "poolmgr")
    assert_exit(proc, 2)
    assert "gateway" in proc.stderr.lower(), ran(proc)
    recs = box.log_records()
    assert [(r["step"], r["status"]) for r in recs] == [("apply", "refused")], recs
    assert "gateway" in str(recs[0].get("detail")).lower(), recs
    assert s3_calls(box) == [], f"a refused apply fetched from S3: {s3_calls(box)}"
    written = set(files_under(box.root))
    assert written == {"var/lib/loom/deploy.log"}, f"a refused apply wrote {sorted(written)}"


def test_only_poolmgr_from_another_release_fetches_and_writes_nothing(box: Box) -> None:
    """A12, §6.12: after release A, `apply --release B --only poolmgr` is refused by P4
    before any fetch or write: no `aws s3` call, the files unchanged but for one new
    deploy-log record, `apply`/`refused`."""
    box.publish(SHA_A)
    first = box.deploy("apply", "--release", SHA_A)
    assert_exit(first, 0)
    box.publish(SHA_B)
    log = box.var_lib / "deploy.log"
    before = files_under(box.root)
    log_before = log.read_text()
    n_s3, n_log = len(s3_calls(box)), len(box.log_records())

    proc = box.deploy("apply", "--release", SHA_B, "--only", "poolmgr")
    assert_exit(proc, 2)
    new = box.log_records()[n_log:]
    assert [(r["step"], r["status"]) for r in new] == [("apply", "refused")], new
    assert "gateway" in str(new[0].get("detail")).lower(), new
    assert s3_calls(box)[n_s3:] == [], f"a refused apply fetched: {s3_calls(box)[n_s3:]}"
    after = files_under(box.root)
    assert log.read_text().startswith(log_before), "the deploy log was rewritten, not appended"
    key = "var/lib/loom/deploy.log"
    assert set(after) == set(before), f"files added or removed: {set(after) ^ set(before)}"
    changed = sorted(k for k in after if k != key and after[k] != before[k])
    assert changed == [], f"a refused apply changed {changed}"


# ---------------------------------------------------------------- 2. P4 in plan_steps


@pytest.mark.parametrize(("running", "target"), [(LO, HI), (HI, LO)], ids=["up", "down"])
def test_only_poolmgr_to_another_release_is_refused_in_either_sha_order(
    running: str, target: str
) -> None:
    """A12, §6.6 P4, §6.12 ("newer" means "another release"): refused whichever way the two
    shas sort, from a plain manifest and from one with `running`."""
    d = deploy()
    plain = manifest(running)
    with_running = {**manifest(running), "running": dict.fromkeys(ORDER, running)}
    for current in (plain, with_running):
        refusal(lambda c=current: d.plan_steps(c, manifest(target), ["poolmgr"]))


@pytest.mark.parametrize(("gateway", "target"), [(LO, HI), (HI, LO)], ids=["up", "down"])
def test_a_poolmgr_behind_the_gateway_may_catch_up_but_not_move_away(
    gateway: str, target: str
) -> None:
    """§6.6 P4: the gateway runs `gateway`, the poolmgr the other release. `--only poolmgr`
    to the gateway's release is allowed; to any other release it is refused."""
    d = deploy()
    current = {
        **manifest(gateway),
        "running": {**dict.fromkeys(ORDER, gateway), "poolmgr": target},
    }
    steps = d.plan_steps(current, manifest(gateway), ["poolmgr"])
    assert plan_ups(steps) == ["poolmgr"], steps
    third = "7" * 40
    refusal(lambda: d.plan_steps(current, manifest(third), ["poolmgr"]))


@pytest.mark.parametrize(("running", "target"), [(LO, HI), (HI, LO)], ids=["up", "down"])
def test_gateway_and_poolmgr_together_are_not_refused_in_either_sha_order(
    running: str, target: str
) -> None:
    d = deploy()
    current = manifest(running)
    for only in (None, ["gateway", "poolmgr"]):
        steps = d.plan_steps(current, manifest(target), only)
        names = plan_ups(steps)
        assert names.index("gateway") < names.index("poolmgr"), steps


@pytest.mark.parametrize("target", [LO, HI])
def test_only_poolmgr_while_the_gateway_runs_nothing_is_refused(target: str) -> None:
    """§6.6 P4: before the first deploy (current None, or a record where no gateway runs)
    the poolmgr alone would run a release the gateway does not: a `Refused` naming the
    gateway (not a TypeError), which never says the gateway runs release `None`."""
    d = deploy()
    no_gateway = {
        **manifest(HI),
        "running": {s: HI for s in ORDER if s != "gateway"},
    }
    for current in (None, no_gateway):
        msg = refusal(lambda c=current: d.plan_steps(c, manifest(target), ["poolmgr"]))
        assert "None" not in msg, f"the refusal names a release None: {msg}"


# ---------------------------------------------------------------- 3. redis only if changed


@pytest.mark.parametrize(
    ("old", "new"), [(REDIS_LO, REDIS_HI), (REDIS_HI, REDIS_LO)], ids=["up", "down"]
)
def test_redis_is_updated_when_its_image_changed_either_way(old: str, new: str) -> None:
    """A12, §6.6 step 4: redis (first) iff its image changed, whichever way the refs sort."""
    d = deploy()
    steps = d.plan_steps(manifest(SHA_A, redis=old), manifest(SHA_B, redis=new))
    assert plan_ups(steps) == ORDER, steps


@pytest.mark.parametrize("image", [REDIS_LO, REDIS_HI])
def test_redis_is_not_updated_when_its_image_is_unchanged(image: str) -> None:
    d = deploy()
    steps = d.plan_steps(manifest(SHA_A, redis=image), manifest(SHA_B, redis=image))
    assert plan_ups(steps) == [s for s in ORDER if s != "redis"], steps


# ---------------------------------------------------------------- 4. m3-p's host half


class FakeCompose:
    """`started_at("poolmgr")` before and after the refused apply."""

    def __init__(self, before: str | None, after: str | None) -> None:
        self.values = [before, after]
        self.asked: list[str] = []

    def started_at(self, service: str) -> str | None:
        self.asked.append(service)
        return self.values.pop(0) if self.values else None


@dataclass
class P4Case:
    code: int = 2
    detail: str = "P4: poolmgr would run release b2b2 while the gateway runs release a1a1"
    before: str | None = "2026-11-18T00:00:01.000000000Z"
    after: str | None = "2026-11-18T00:00:01.000000000Z"


def run_p4_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: P4Case
) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    d = deploy()
    monkeypatch.setenv("LOOM_DEPLOY_ROOT", str(tmp_path))
    paths = d.Paths()
    paths.var_lib.mkdir(parents=True, exist_ok=True)
    calls: list[tuple[str, Any]] = []

    def fake_apply(sha: str, only: Any) -> int:
        calls.append((sha, only))
        rec = {"ts": "2026-11-18T00:00:02Z", "release": sha, "step": "apply"}
        rec |= {"status": "refused" if case.code == 2 else "failed", "detail": case.detail}
        with paths.deploy_log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return case.code

    monkeypatch.setattr(d, "apply_release", fake_apply)
    result = d.check_p4_refusal(paths, FakeCompose(case.before, case.after), SHA_A, SHA_B)
    assert isinstance(result, dict) and isinstance(result.get("ok"), bool), result
    return result, calls


def test_m3p_host_half_passes_a_refusal_naming_the_gateway(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, calls = run_p4_check(monkeypatch, tmp_path, P4Case())
    assert result["ok"] is True, result
    assert calls == [(SHA_B, ["poolmgr"])], calls


@pytest.mark.parametrize("code", [0, 1, 3])
def test_m3p_host_half_fails_when_the_apply_exits_other_than_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: int
) -> None:
    """§6.12: m3-p's refusal must exit 2 (refused); 0 (done) or 1 (failed) fail m3-p."""
    result, _ = run_p4_check(monkeypatch, tmp_path, P4Case(code=code))
    assert result["ok"] is False, result


def test_m3p_host_half_fails_when_the_refusal_does_not_name_the_gateway(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = P4Case(detail="P4: poolmgr would run another release than the one running")
    result, _ = run_p4_check(monkeypatch, tmp_path, case)
    assert result["ok"] is False, result


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("2026-11-18T00:00:01.000000000Z", "2026-11-18T00:00:02.000000000Z"),
        ("2026-11-18T00:00:02.000000000Z", "2026-11-18T00:00:01.000000000Z"),
        (None, "2026-11-18T00:00:01.000000000Z"),
    ],
    ids=["later", "earlier", "appeared"],
)
def test_m3p_host_half_fails_when_the_poolmgr_container_changed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, before: str | None, after: str | None
) -> None:
    """§6.12: the refused apply must not recreate the poolmgr container."""
    result, _ = run_p4_check(monkeypatch, tmp_path, P4Case(before=before, after=after))
    assert result["ok"] is False, result


# ---------------------------------------------------------------- 5. the site parameter


@pytest.mark.parametrize("port", [1, 5432, 65535])
def test_validate_site_accepts_ports_in_range(port: int) -> None:
    deploy().validate_site(site(db_port=port))


@pytest.mark.parametrize("port", [0, -1, 65536, True, False], ids=repr)
def test_validate_site_refuses_a_port_out_of_range_or_a_bool(port: Any) -> None:
    """§6.12: `db_port` is a JSON number that is a port (a JSON `true` is not)."""
    d = deploy()
    with pytest.raises(d.Refused):
        d.validate_site(site(db_port=port))


def test_validate_site_brand_name_length_limit() -> None:
    """The brand name is the display name of `EMAIL_FROM` (§6.11): 64 characters pass, 65
    are refused."""
    d = deploy()
    d.validate_site(site(brand_name="Q" * 64))
    with pytest.raises(d.Refused):
        d.validate_site(site(brand_name="Q" * 65))


@pytest.mark.parametrize("char", list(',;:@()[]<>"\\'))
def test_validate_site_refuses_a_brand_name_with_a_mail_special(char: str) -> None:
    """§6.11: `EMAIL_FROM` is `<name> <no-reply@<domain>>`, so the name holds none of the
    characters RFC 5322 treats as special in a display name."""
    d = deploy()
    with pytest.raises(d.Refused):
        d.validate_site(site(brand_name=f"Quill{char}work"))


@pytest.mark.parametrize("char", ["\x00", "\t", "\n", "\r", "\x1f"], ids=repr)
def test_validate_site_refuses_a_brand_name_with_a_control_character(char: str) -> None:
    d = deploy()
    with pytest.raises(d.Refused):
        d.validate_site(site(brand_name=f"Quill{char}work"))


def test_validate_site_accepts_a_plain_brand_name() -> None:
    doc = site(brand_name="Quill-work Test 2")
    assert deploy().validate_site(doc) == doc


# ---------------------------------------------------------------- 6. D10 schemes


@pytest.mark.parametrize(
    "scheme", ["redis", "rediss", "sqlite", "sqlserver", "zz", "postgresqlx", "mysql", "https"]
)
def test_check_database_url_refuses_other_schemes(scheme: str) -> None:
    """D10, §6.12: only a postgresql:// URL passes, whatever its scheme sorts against."""
    d = deploy()
    url = f"{scheme}://loom_app:pw@db:5432/loom_platform{RDS_QUERY}"
    with pytest.raises(d.Refused):
        d.check_database_url(url)


def test_check_database_url_still_accepts_postgresql() -> None:
    url = f"postgresql://loom_app:pw@db:5432/loom_platform{RDS_QUERY}"
    assert deploy().check_database_url(url) is None


# ---------------------------------------------------------------- 7. env-file values


@pytest.mark.parametrize("value", ["\n", "\r", "\r\n", "\0", "a\nb", "it's\n"], ids=repr)
def test_an_env_value_with_a_line_break_or_nul_is_refused(value: str) -> None:
    """A13, §6.12: an env-file value cannot hold a line break (compose would read another
    line) or a NUL (no environment can): rendering it fails instead of writing it."""
    d = deploy()
    with pytest.raises((d.DeployError, d.Refused)):
        d.quote_env(value)
    with pytest.raises((d.DeployError, d.Refused)):
        d.env_text({"LOOM_X": value})


@pytest.mark.parametrize(
    "value",
    [
        "it's",
        "'",
        'it\'s "quoted"',
        "it's C:\\new\\tab \\$HOME",
        "it's $HOME ${HOME} $$ $",
        "'\\\"'",
        'p@ss "w" C:\\new $HOME ${X} $$',
        "",
        "plain-token_123",
    ],
    ids=repr,
)
def test_env_values_read_back_exactly_as_compose_reads_them(value: str) -> None:
    """A13, §6.12: an env file holds the value literally (no `$` interpolation, no escape
    turned into another character), with or without a single quote in it."""
    text = deploy().env_text({"LOOM_X": value, "LOOM_Y": "after"})
    assert compose_env_file(text) == {"LOOM_X": value, "LOOM_Y": "after"}, text


# ---------------------------------------------------------------- 8, 9. re-rendering


@dataclass
class Rerendered:
    box: Box
    first: Any
    second: Any
    google: dict[str, str]
    web_env_first: dict[str, str]
    web_env_second: dict[str, str]
    gateway_secrets: set[str]


STALE_FILE = "stale-left-over.pem"
STALE_DIR = "stale-left-over.d"


@pytest.fixture(scope="module")
def rerendered() -> Iterator[Rerendered]:
    """Release A applied with an `auth` secret without the Google keys; then the Google
    keys added and a stale file and directory planted in the gateway's secret directory;
    then release A applied again."""
    store = full_store()
    auth = json.loads(str(store["auth"]))
    store["auth"] = json.dumps({"auth_secret": auth["auth_secret"]})
    box = Box(deploy_root_base(), store=store, site_doc=site())
    try:
        box.publish(SHA_A)
        first = box.deploy("apply", "--release", SHA_A)
        web = box.run_loom / "env" / "web.env"
        env1 = compose_env_file(web.read_text()) if first.returncode == 0 else {}
        box.store["auth"] = json.dumps(auth)
        box.save_store()
        d = box.run_loom / "secrets" / "gateway"
        if d.is_dir():
            d.chmod(0o700)
            (d / STALE_FILE).write_text("old\n")
            (d / STALE_DIR).mkdir()
            (d / STALE_DIR / "inner").write_text("old\n")
            # A rendered file made newer than the stale entries: they then sit between
            # rendered files in the directory's listing order, whichever way it runs.
            kept = d / "redis-ca.crt"
            if kept.is_file():
                text = kept.read_text()
                kept.unlink()
                kept.write_text(text)
                kept.chmod(0o400)
            d.chmod(0o500)
        second = box.deploy("apply", "--release", SHA_A)
        env2 = compose_env_file(web.read_text()) if second.returncode == 0 else {}
        names = {p.name for p in d.iterdir()} if d.is_dir() else set()
        google = {
            "AUTH_GOOGLE_ID": auth["google_id"],
            "AUTH_GOOGLE_SECRET": auth["google_secret"],
        }
        yield Rerendered(box, first, second, google, env1, env2, names)
    finally:
        box.close()


def test_web_has_no_google_settings_when_the_auth_secret_lacks_them(
    rerendered: Rerendered,
) -> None:
    """§9, §3: Google sign-in is off when its keys are unset: no `AUTH_GOOGLE_*` for web."""
    assert_exit(rerendered.first, 0)
    env = rerendered.web_env_first
    assert "AUTH_SECRET" in env, sorted(env)
    present = sorted(k for k in env if k.startswith("AUTH_GOOGLE"))
    assert present == [], f"web.env has {present} with no Google keys in the auth secret"


def test_web_gets_the_google_settings_when_the_auth_secret_has_them(
    rerendered: Rerendered,
) -> None:
    """§9: `AUTH_GOOGLE_ID`/`AUTH_GOOGLE_SECRET` from the `auth` secret's `google_id` and
    `google_secret`, rendered at the next apply."""
    assert_exit(rerendered.second, 0)
    env = rerendered.web_env_second
    for k, v in rerendered.google.items():
        assert env.get(k) == v, f"web.env {k} is {env.get(k)!r}"


def test_a_rerender_removes_stale_entries_from_a_secret_directory(
    rerendered: Rerendered,
) -> None:
    """A13, §6.12: a service's secret directory holds the files it reads; a re-render
    removes anything else left in it."""
    assert_exit(rerendered.second, 0)
    names = rerendered.gateway_secrets
    assert STALE_FILE not in names, f"a stale file survived the re-render: {sorted(names)}"
    assert STALE_DIR not in names, f"a stale directory survived the re-render: {sorted(names)}"
    assert {"redis-ca.crt", "pod-ca.crt"} <= names, sorted(names)
