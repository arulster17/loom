"""`loom-deploy apply` against fakes (M3 spec §6.3, §6.6, §6.11, §6.12; A13, A29, A35).

`loom-deploy` runs as on the box: `python -S -m loom_devstack.deploy` with only
`loom_devstack` on PYTHONPATH, `LOOM_DEPLOY_ROOT` a scratch directory under /dev/shm (a
tmpfs), and fake `aws`, `docker` and `chown` first on PATH (`devstack_m3_kit.py`). The fake
docker answers what §6.12 says `apply` calls (`login`, `compose pull`, `compose run --rm
migrate`, `compose up -d --no-deps <svc>`, `compose ps -q <svc>`, `inspect`) and starts
nothing. The release's bundle is the repository's `deploy/` directory, uploaded as the
image workflow would; the refusal and manifest tests use a two-file stand-in (nothing is
rendered or started before them, so its content does not matter).

Readings this file takes where §6.12 leaves a choice (the narrowest one):
- `LOOM_CADDY_TLS` (`ops_email` on the box) is rendered into `env/caddy.env` with
  `LOOM_DOMAIN` (compose.env holds only `LOOM_DEPLOY_ROOT` and the image variables);
- env files exist for caddy, web, gateway, billing, poolmgr, migrate and acceptance, and
  secrets directories for gateway, billing, web, poolmgr, redis, grafana and acceptance;
  each Redis client (gateway, billing, poolmgr, web, acceptance) has its own
  `redis-ca.crt` (§6.5's `ssl_ca_certs=/run/loom/secrets/<svc>/redis-ca.crt`);
- reading the secrets is part of "render", so it comes after the manifest;
- a refused apply's `apply`/`refused` record is the only refused record and nothing
  of render, pull, migrate, update or record is logged; other records before it (an
  `apply`/`start`, the manifest) are not forbidden;
- `release.json` is JSON that names the release's sha;
- the deploy log is "printed": each record also appears as a JSON line on stdout or
  stderr.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import os
import re
import time
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from devstack_m3_kit import (
    IMAGE_VARS,
    ORDER,
    REGISTRY,
    ROOT,
    UIDS,
    VOLUMES,
    Box,
    aws_op,
    chowns,
    compose_calls,
    compose_target,
    deploy_root_base,
    ecr_images,
    field,
    mode,
    mount_fstype,
    option_values,
    owner_of,
    read_env_file,
    runs,
    secret_values,
    site,
    text_files,
    ups,
)

SHA_A = "a1" * 20
SHA_B = "b2" * 20
DOMAIN = "unit.example.test"
NAME = "Quillwork Test"
OPS = "ops@unit.example.test"
ALWAYS = [s for s in ORDER if s != "redis"]
ENV_SERVICES = ("caddy", "web", "gateway", "billing", "poolmgr", "migrate", "acceptance")
SECRET_SERVICES = ("gateway", "billing", "web", "poolmgr", "redis", "grafana", "acceptance")
REDIS_CLIENTS = ("gateway", "billing", "poolmgr", "web", "acceptance")
STEPS = ("apply", "manifest", "render", "pull", "migrate", "update", "record")
STATUSES = ("start", "ok", "failed", "refused")
STAND_IN = {"compose.prod.yml": b"services: {}\n", "Caddyfile": b"# stand-in\n"}


def unit_site(**over: Any) -> dict[str, Any]:
    return site(**{"domain": DOMAIN, "brand_name": NAME, "ops_email": OPS, **over})


def assert_exit(proc: Any, code: int) -> None:
    """`loom-deploy` ran (an import error also exits 1) and exited with `code`."""
    assert "No module named" not in proc.stderr, ran(proc)
    assert proc.returncode == code, ran(proc)


def ran(proc: Any) -> str:
    out, err = proc.stdout[-3000:], proc.stderr[-3000:]
    return f"exit {proc.returncode}\n--- stdout\n{out}\n--- stderr\n{err}"


@dataclass
class Applied:
    box: Box
    proc: Any
    override: Path
    error: str | None = None

    def need(self) -> Box:
        assert self.error is None, self.error
        assert self.proc.returncode == 0, ran(self.proc)
        return self.box


@pytest.fixture(scope="module")
def applied() -> Iterator[Applied]:
    """One first apply of release A (no release.json before), with AWS_REGION=eu-west-1
    and a compose override; the read-only checks below share it."""
    base = deploy_root_base()
    box = Box(base, site_doc=unit_site())
    override = base / "override.yml"
    override.write_text("services: {}\n")
    try:
        box.publish(SHA_A)
    except AssertionError as e:
        yield Applied(box, None, override, error=str(e))
        box.close()
        return
    proc = box.deploy(
        "apply",
        "--release",
        SHA_A,
        AWS_REGION="eu-west-1",
        LOOM_DEPLOY_COMPOSE_OVERRIDE=str(override),
    )
    yield Applied(box, proc, override)
    box.close()


@pytest.fixture
def box() -> Iterator[Box]:
    b = Box(deploy_root_base(), site_doc=unit_site())
    yield b
    b.close()


def deploy_module() -> Any:
    return importlib.import_module("loom_devstack.deploy")


# ---------------------------------------------------------------- the files apply writes


def test_apply_exits_0_and_records_the_release(applied: Applied) -> None:
    box = applied.need()
    rel = box.var_lib / "release.json"
    assert rel.is_file(), "no <root>/var/lib/loom/release.json"
    doc = json.loads(rel.read_text())
    assert SHA_A in json.dumps(doc), f"release.json does not name {SHA_A}: {doc}"


def test_the_bundle_is_extracted_and_compose_runs_from_it(applied: Applied) -> None:
    box = applied.need()
    rel = box.var_lib / "releases" / SHA_A
    compose = rel / "compose.prod.yml"
    assert compose.is_file(), f"the bundle is not extracted to {rel}"
    assert compose.read_bytes() == (ROOT / "deploy" / "compose.prod.yml").read_bytes()
    assert (rel / "Caddyfile").is_file(), "the bundle's Caddyfile is not extracted beside it"
    tl = box.timeline()
    calls = compose_calls(tl)
    assert calls, "loom-deploy ran no `docker compose`"
    # §6.12 fixes the project and the files, not how they are passed: `-p` or
    # COMPOSE_PROJECT_NAME; `-f` (absolute or relative to the cwd) or COMPOSE_FILE.
    for n, opts, sub, _ in calls:
        project, files = compose_target(tl[n], opts)
        assert files == [compose, applied.override], (
            f"compose {sub}: files {files}, not the release's file then "
            "LOOM_DEPLOY_COMPOSE_OVERRIDE"
        )
        cwd = Path(tl[n].get("cwd") or "/")
        env_files = [Path(os.path.normpath(cwd / f)) for f in option_values(opts, "--env-file")]
        assert env_files == [box.run_loom / "compose.env"], opts
        assert project == "loom", f"compose {sub}: project {project!r}, not the default `loom`"


def test_compose_env_holds_the_root_and_the_image_variables(applied: Applied) -> None:
    box = applied.need()
    path = box.run_loom / "compose.env"
    assert path.is_file(), "no <root>/run/loom/compose.env"
    assert mode(path) == 0o400, f"compose.env mode {oct(mode(path))}"
    env = read_env_file(path)
    assert env.get("LOOM_DEPLOY_ROOT") == str(box.root), env.get("LOOM_DEPLOY_ROOT")
    images = box.releases[SHA_A]["images"]
    for var, svc in IMAGE_VARS.items():
        assert env.get(var) == images[svc], f"{var}={env.get(var)!r}, manifest {images[svc]!r}"
    assert owner_of(path, chowns(box.timeline())) is None, "compose.env was chowned"


def test_env_files_are_0400_and_never_chowned(applied: Applied) -> None:
    box = applied.need()
    env_dir = box.run_loom / "env"
    for svc in ENV_SERVICES:
        assert (env_dir / f"{svc}.env").is_file(), f"no env/{svc}.env"
    recorded = chowns(box.timeline())
    for p in env_dir.iterdir():
        assert p.is_file() and p.suffix == ".env", f"unexpected {p.name} in run/loom/env"
        assert mode(p) == 0o400, f"{p.name} mode {oct(mode(p))}"
        assert owner_of(p, recorded) is None, f"{p.name} was chowned (compose reads it as root)"


def test_secrets_are_0400_in_0500_directories_owned_by_the_image_uid(applied: Applied) -> None:
    box = applied.need()
    sec = box.run_loom / "secrets"
    recorded = chowns(box.timeline())
    for svc in SECRET_SERVICES:
        assert (sec / svc).is_dir(), f"no run/loom/secrets/{svc}/"
    for d in sorted(sec.iterdir()):
        svc = d.name
        assert svc in UIDS, f"secrets for {svc}, which has no uid in §6.12's table"
        want = f"{UIDS[svc]}:{UIDS[svc]}"
        files = [p for p in d.rglob("*") if p.is_file()]
        assert files, f"secrets/{svc}/ is empty"
        for p in [d, *[q for q in d.rglob("*") if q.is_dir()]]:
            assert mode(p) == 0o500, f"{p.relative_to(box.root)} mode {oct(mode(p))}"
            assert owner_of(p, recorded) == want, f"{p.relative_to(box.root)} not chowned {want}"
        for p in files:
            assert mode(p) == 0o400, f"{p.relative_to(box.root)} mode {oct(mode(p))}"
            assert owner_of(p, recorded) == want, f"{p.relative_to(box.root)} not chowned {want}"


def test_each_service_gets_the_secret_files_it_reads(applied: Applied) -> None:
    box = applied.need()
    store = box.store
    sec = box.run_loom / "secrets"
    ca = field(store, "redis", "ca_crt").strip()
    for svc in REDIS_CLIENTS:
        p = sec / svc / "redis-ca.crt"
        assert p.is_file(), f"no secrets/{svc}/redis-ca.crt (§6.5)"
        assert p.read_text().strip() == ca, f"secrets/{svc}/redis-ca.crt is not the Redis CA"
    hook = sec / "acceptance" / "webhook_secret"
    assert hook.is_file(), "no secrets/acceptance/webhook_secret (§6.12)"
    assert hook.read_text().strip() == field(store, "stripe", "webhook_secret")

    def texts(svc: str) -> list[str]:
        return list(text_files(sec / svc).values())

    pw = field(store, "grafana", "admin_password")
    assert pw in [t.strip() for t in texts("grafana")], "no grafana file holds its admin password"
    assert any(field(store, "redis", "password") in t for t in texts("redis")), (
        "no file of secrets/redis/ holds requirepass's password"
    )
    key_line = field(store, "redis", "server_key").splitlines()[1]
    assert any(key_line in t for t in texts("redis")), "Redis's server key is not rendered"
    billing = [*texts("billing"), (box.run_loom / "env" / "billing.env").read_text()]
    for k in ("secret_key", "webhook_secret"):
        assert any(field(store, "stripe", k) in t for t in billing), f"billing lacks stripe {k}"


def test_the_volume_directories_are_created_and_chowned(applied: Applied) -> None:
    box = applied.need()
    recorded = chowns(box.timeline())
    for sub, uid in VOLUMES.items():
        p = box.var_lib / sub
        assert p.is_dir(), f"apply did not create <root>/var/lib/loom/{sub}"
        assert owner_of(p, recorded) == f"{uid}:{uid}", f"var/lib/loom/{sub} not chowned {uid}"


def test_the_site_settings_are_rendered_into_the_env_files(applied: Applied) -> None:
    """§6.11's table, from the site parameter (A35)."""
    box = applied.need()

    def env(svc: str) -> dict[str, str]:
        return read_env_file(box.run_loom / "env" / f"{svc}.env")

    caddy, web, gateway, billing = env("caddy"), env("web"), env("gateway"), env("billing")
    assert caddy.get("LOOM_DOMAIN") == DOMAIN, caddy
    assert caddy.get("LOOM_CADDY_TLS") == OPS, "caddy's tls argument is not ops_email"
    assert web.get("LOOM_BRAND_NAME") == NAME
    assert web.get("AUTH_URL") == f"https://{DOMAIN}"
    assert web.get("LOOM_PUBLIC_API_URL") == f"https://api.{DOMAIN}/v1"
    assert web.get("EMAIL_FROM") == f"{NAME} <no-reply@{DOMAIN}>", web.get("EMAIL_FROM")
    assert gateway.get("LOOM_CORS_ORIGINS") == f"https://{DOMAIN}"
    assert billing.get("LOOM_BRAND_NAME") == NAME


def db_url_parts(url: str) -> tuple[urllib.parse.SplitResult, dict[str, list[str]]]:
    u = urllib.parse.urlsplit(url)
    return u, urllib.parse.parse_qs(u.query)


def assert_rds_url(url: str, user: str, password: str, what: str) -> None:
    u, q = db_url_parts(url)
    assert u.scheme.startswith("postgresql"), f"{what}: scheme {u.scheme}"
    assert u.username == user, f"{what}: user {u.username}, not {user}"
    assert urllib.parse.unquote(u.password or "") == password, f"{what}: wrong password"
    assert u.hostname == site()["db_host"] and u.port == 5432, f"{what}: {u.hostname}:{u.port}"
    assert u.path == "/loom_platform", f"{what}: database {u.path}"
    assert q.get("sslmode") == ["verify-full"], f"{what}: sslmode {q.get('sslmode')}"
    assert q.get("sslrootcert") == ["/etc/loom/rds-ca.pem"], f"{what}: {q.get('sslrootcert')}"


def test_database_urls_are_verify_full_with_the_right_roles(applied: Applied) -> None:
    """§6.4, §6.12: the app URL (`loom_app`, `app-db-password`) for the services and the
    acceptance, the owner URL (the RDS master secret) only in migrate's env file; every
    rendered URL passes `check_database_url` (D10)."""
    box = applied.need()
    store = box.store
    app_pw = str(store["app-db-password"])
    master = json.loads(str(store[site()["db_master_secret_arn"]]))
    env_dir = box.run_loom / "env"
    web = read_env_file(env_dir / "web.env")
    assert_rds_url(web.get("DATABASE_URL", ""), "loom_app", app_pw, "web DATABASE_URL")
    acc = read_env_file(env_dir / "acceptance.env")
    assert_rds_url(acc.get("LOOM_ACCEPT_DATABASE_URL", ""), "loom_app", app_pw, "acceptance")
    mig = read_env_file(env_dir / "migrate.env")
    owner = [v for v in mig.values() if v.startswith("postgres")]
    assert len(owner) == 1, f"migrate.env holds {len(owner)} database URLs"
    assert_rds_url(owner[0], master["username"], master["password"], "migrate's owner URL")
    assert mig.get("LOOM_APP_DATABASE_PASSWORD") == app_pw
    for name, text in text_files(box.run_loom).items():
        for url in re.findall(r"postgres(?:ql)?(?:\+\w+)?://\S+", text):
            url = url.strip("\"'")
            _, q = db_url_parts(url)
            assert q.get("sslmode") == ["verify-full"], f"{name.name}: {q.get('sslmode')}"
            assert q.get("sslrootcert") == ["/etc/loom/rds-ca.pem"], name.name
            assert u_user(url) != master["username"] or name.name == "migrate.env", (
                f"the owner URL is rendered outside migrate's env file ({name.name})"
            )
            deploy_module().check_database_url(url)


def u_user(url: str) -> str | None:
    return urllib.parse.urlsplit(url).username


def test_the_acceptance_service_gets_its_inputs(applied: Applied) -> None:
    box = applied.need()
    acc = read_env_file(box.run_loom / "env" / "acceptance.env")
    u = urllib.parse.urlsplit(acc.get("LOOM_ACCEPT_REDIS_URL", ""))
    assert u.scheme == "rediss", f"LOOM_ACCEPT_REDIS_URL scheme {u.scheme!r}"
    assert (u.hostname, u.port, u.path) == ("redis", 6379, "/0"), u
    assert urllib.parse.unquote(u.password or "") == field(box.store, "redis", "password")
    q = urllib.parse.parse_qs(u.query)
    assert q.get("ssl_ca_certs") == ["/run/loom/secrets/acceptance/redis-ca.crt"], q


def test_no_secret_value_outside_run_loom(applied: Applied) -> None:
    """Not on a command line (aws, docker, chown), not on stdout/stderr, not in the deploy
    log, compose.env, the extracted bundle, release.json or any file outside run/loom's
    env and secrets directories (A13)."""
    box = applied.need()
    values = [*secret_values(box.store), box.ecr_token]
    argv = box.all_argv()
    output = applied.proc.stdout + applied.proc.stderr
    files = text_files(box.var_lib, box.home, box.tmp, box.work)
    files[box.run_loom / "compose.env"] = (box.run_loom / "compose.env").read_text()
    for v in values:
        assert v not in argv, "a secret value is on a command line"
        assert v not in output, "loom-deploy printed a secret value"
        for path, text in files.items():
            assert v not in text, f"a secret value is in {path.relative_to(box.base)}"
    runpod = str(box.store["serving-runpod-key"])
    every = text_files(box.root)
    assert not any(runpod in t for t in every.values()), "the serving RunPod key was rendered"


def test_the_serving_runpod_key_is_never_read(applied: Applied) -> None:
    box = applied.need()
    named = [a for a in box.calls("aws") if "serving-runpod-key" in " ".join(a)]
    assert not named, f"apply named serving-runpod-key: {named}"


def test_every_aws_call_names_the_region(applied: Applied) -> None:
    box = applied.need()
    calls = box.calls("aws")
    assert calls, "apply made no aws call"
    for argv in calls:
        assert option_values(argv, "--region") == ["eu-west-1"], f"aws {argv[:2]}: no --region"


def test_ecr_login_takes_the_password_on_stdin(applied: Applied) -> None:
    box = applied.need()
    assert any(aws_op(a) == ["ecr", "get-login-password"] for a in box.calls("aws"))
    logins = [r for r in box.timeline() if r["tool"] == "docker" and r["argv"][:1] == ["login"]]
    assert logins, "no docker login"
    for r in logins:
        argv = r["argv"]
        assert "--password-stdin" in argv and option_values(argv, "--username", "-u") == ["AWS"]
        assert argv[-1] == REGISTRY, f"docker login to {argv[-1]}, not the manifest's registry"
        assert r.get("stdin_is_token"), "docker login did not get the ECR password on stdin"


def first(timeline: list[dict[str, Any]], pred: Any, what: str) -> int:
    for n, r in enumerate(timeline):
        if pred(r):
            return n
    raise AssertionError(f"no {what} call")


def test_the_steps_run_in_order(applied: Applied) -> None:
    """manifest -> render -> ECR login -> pull -> migrate -> the updates of §6.6 -> record;
    each update waits for its service's health before the next (§6.12)."""
    box = applied.need()
    tl = box.timeline()
    secrets_dir = box.run_loom / "secrets"
    manifest = first(
        tl,
        lambda r: r["tool"] == "aws" and any(f"releases/{SHA_A}.json" in a for a in r["argv"]),
        "manifest download",
    )
    read = first(
        tl,
        lambda r: (
            r["tool"] == "aws" and aws_op(r["argv"]) == ["secretsmanager", "get-secret-value"]
        ),
        "secret read",
    )
    render = first(
        tl,
        lambda r: (
            r["tool"] == "chown"
            and any(Path(a).is_relative_to(secrets_dir) for a in r["argv"] if a.startswith("/"))
        ),
        "chown of a rendered secret",
    )
    login = first(tl, lambda r: r["tool"] == "docker" and r["argv"][:1] == ["login"], "login")
    calls = compose_calls(tl)
    pulls = [n for n, _, sub, _ in calls if sub == "pull"]
    migrates = [n for n, svc, _ in runs(tl) if svc == "migrate"]
    assert pulls and migrates, f"pull {pulls}, run migrate {migrates}"
    up = ups(tl)
    assert [s for _, s in up] == ORDER, f"updates {[s for _, s in up]}, not {ORDER}"
    assert manifest < read and manifest < render < login < pulls[0] < migrates[0] < up[0][0], (
        f"order: manifest {manifest}, secrets {read}, render {render}, login {login}, "
        f"pull {pulls[0]}, migrate {migrates[0]}, first up {up[0][0]}"
    )
    marks = [n for n, _ in up] + [len(tl)]
    for (n, svc), nxt in zip(up, marks[1:], strict=True):
        inspected = [
            r
            for r in tl[n + 1 : nxt]
            if r["tool"] == "docker" and "inspect" in r["argv"][:2] and f"cid-{svc}" in r["argv"]
        ]
        assert inspected, f"{svc}: no health inspect between its up and the next step"


def test_the_deploy_log_is_json_lines_in_step_order(applied: Applied) -> None:
    box = applied.need()
    recs = box.log_records()
    assert recs, "no records in <root>/var/lib/loom/deploy.log"
    for r in recs:
        assert {"ts", "release", "step", "status"} <= set(r), f"record lacks keys: {r}"
        assert r["release"] == SHA_A, r
        assert r["step"] in STEPS and r["status"] in STATUSES, r
        ts = dt.datetime.fromisoformat(str(r["ts"]).replace("Z", "+00:00"))
        assert ts.utcoffset() == dt.timedelta(0), f"ts not UTC: {r['ts']}"
        if r["step"] == "update":
            assert r.get("service") in ORDER, r
        assert r["status"] != "failed" and r["status"] != "refused", r
    assert (recs[0]["step"], recs[0]["status"]) == ("apply", "start"), recs[0]
    assert "only" in recs[0] and recs[0]["only"] is None, recs[0]
    firsts = [next(i for i, r in enumerate(recs) if r["step"] == s) for s in STEPS[1:]]
    assert firsts == sorted(firsts), f"steps out of order: {[r['step'] for r in recs]}"
    for s in ("manifest", "render", "pull", "migrate", "record"):
        assert any(r["step"] == s and r["status"] == "ok" for r in recs), f"no {s}/ok"
    updates = [(r["service"], r["status"]) for r in recs if r["step"] == "update"]
    assert updates == [(s, st) for s in ORDER for st in ("start", "ok")], updates
    printed = []
    for line in (applied.proc.stdout + "\n" + applied.proc.stderr).splitlines():
        try:
            printed.append(json.loads(line))
        except ValueError:
            continue
    missing = [r for r in recs if r not in printed]
    assert not missing, f"deploy-log records not printed: {missing[:3]}"


# ---------------------------------------------------------------- other applies


def test_the_region_defaults_to_us_east_1(box: Box) -> None:
    box.publish(SHA_A)
    proc = box.deploy("apply", "--release", SHA_A, AWS_REGION=None, AWS_DEFAULT_REGION=None)
    assert proc.returncode == 0, ran(proc)
    for argv in box.calls("aws"):
        assert option_values(argv, "--region") == ["us-east-1"], f"aws {argv[:2]}"


def test_a_second_apply_of_the_release_updates_all_but_redis(box: Box) -> None:
    box.publish(SHA_A)
    first_run = box.deploy("apply", "--release", SHA_A)
    assert first_run.returncode == 0, ran(first_run)
    n = len(box.timeline())
    again = box.deploy("apply", "--release", SHA_A)
    assert again.returncode == 0, ran(again)
    assert [s for _, s in ups(box.timeline()[n:])] == ALWAYS


def test_a_health_timeout_fails_the_update_and_keeps_the_rendered_files(box: Box) -> None:
    box.publish(SHA_A)
    box.docker_cfg["health"] = {"gateway": "starting"}
    box.save_docker()
    t0 = time.monotonic()
    proc = box.deploy("apply", "--release", SHA_A, LOOM_DEPLOY_HEALTH_TIMEOUT_S="2")
    took = time.monotonic() - t0
    assert_exit(proc, 1)
    assert took < 60, f"apply took {took:.0f} s with a 2 s health timeout"
    recs = box.log_records()
    gw = [r["status"] for r in recs if r["step"] == "update" and r.get("service") == "gateway"]
    assert gw == ["start", "failed"], f"gateway update records {gw}"
    assert not any(r["step"] == "update" and r.get("service") == "poolmgr" for r in recs)
    assert not any(r["step"] == "record" and r["status"] == "ok" for r in recs)
    assert [s for _, s in ups(box.timeline())] == ["redis", "billing", "gateway"]
    assert not (box.var_lib / "release.json").exists(), "a failed apply recorded the release"
    for p in ("compose.env", "env/web.env", "secrets/gateway/redis-ca.crt"):
        assert (box.run_loom / p).is_file(), f"run/loom/{p} was removed after the failure"


def test_a_failed_migration_stops_before_any_update(box: Box) -> None:
    box.publish(SHA_A)
    box.docker_cfg["run_exit"] = {"migrate": 3}
    box.save_docker()
    proc = box.deploy("apply", "--release", SHA_A)
    assert_exit(proc, 1)
    assert not ups(box.timeline()), "a service was updated after the migration failed"
    recs = box.log_records()
    assert any(r["step"] == "migrate" and r["status"] == "failed" for r in recs), recs[-3:]
    assert not any(r["step"] == "update" for r in recs)


# ---------------------------------------------------------------- refusals (exit 2)


def assert_refused_and_untouched(box: Box, proc: Any, *, mention: str | None = None) -> None:
    assert_exit(proc, 2)
    assert proc.stderr.strip(), "a refusal gives its reason on stderr"
    recs = box.log_records()
    refused = [r for r in recs if r["status"] == "refused"]
    assert len(refused) == 1, f"{len(refused)} refused records: {recs}"
    assert refused[0]["step"] == "apply" and recs[-1] == refused[0], recs
    assert refused[0].get("detail"), "the refused record gives no reason (detail)"
    if mention:
        said = (str(refused[0].get("detail")) + proc.stderr).lower()
        assert mention in said, f"the refusal does not name {mention}: {refused[0]}"
    assert not any(r["step"] in STEPS[2:] for r in recs), f"steps after a refusal: {recs}"
    tl = box.timeline()
    assert not [r for r in tl if r["tool"] == "chown"], "a refused apply chowned something"
    assert not [r for r in tl if r["tool"] == "docker" and r["argv"][:1] in (["login"],)]
    subs = [sub for _, _, sub, _ in compose_calls(tl)]
    assert not set(subs) & {"pull", "run", "up"}, f"a refused apply ran compose {subs}"
    for p in ("compose.env", "env", "secrets"):
        assert not (box.run_loom / p).exists(), f"a refused apply rendered run/loom/{p}"


def test_refused_when_run_loom_is_not_on_a_tmpfs(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    if mount_fstype(root) == "tmpfs":
        pytest.skip("tmp_path is on a tmpfs here; the refusal needs a disk directory")
    box = Box(deploy_root_base(), root=root, site_doc=unit_site())
    try:
        box.publish(SHA_A, bundle_files=STAND_IN)
        proc = box.deploy("apply", "--release", SHA_A)
        assert_refused_and_untouched(box, proc, mention="tmpfs")
    finally:
        box.close()


@pytest.mark.parametrize(
    "over, key",
    [
        ({"domain": None}, "domain"),
        ({"brand_name": None}, "brand_name"),
        ({"domain": ""}, "domain"),
        ({"brand_name": ""}, "brand_name"),
        ({"db_host": None}, "db_host"),
        ({"ops_email": ""}, "ops_email"),
    ],
)
def test_refused_when_the_site_parameter_lacks_a_key(
    box: Box, over: dict[str, Any], key: str
) -> None:
    """A29, §6.11: an apply whose site parameter lacks the domain or the product name (or
    any key of §6.12, or has it empty) is refused and changes nothing."""
    box.set_site(unit_site(**over))
    box.publish(SHA_A, bundle_files=STAND_IN)
    proc = box.deploy("apply", "--release", SHA_A)
    assert_refused_and_untouched(box, proc, mention=key)


def snapshot(d: Path) -> dict[str, tuple[bytes, int, int]]:
    return {
        str(p.relative_to(d)): (p.read_bytes(), mode(p), p.stat().st_mtime_ns)
        for p in d.rglob("*")
        if p.is_file()
    }


def test_p4_only_poolmgr_from_another_release_is_refused(box: Box) -> None:
    box.publish(SHA_A)
    proc = box.deploy("apply", "--release", SHA_A)
    assert proc.returncode == 0, ran(proc)
    box.publish(SHA_B, ecr_images(SHA_B))
    before_files = snapshot(box.run_loom)
    before_release = (box.var_lib / "release.json").read_bytes()
    n_tl, n_log = len(box.timeline()), len(box.log_records())
    refused = box.deploy("apply", "--only", "poolmgr", "--release", SHA_B)
    assert_exit(refused, 2)
    assert "gateway" in refused.stderr.lower(), (
        f"the refusal does not name the gateway: {ran(refused)}"
    )
    new = box.log_records()[n_log:]
    assert [(r["step"], r["status"]) for r in new if r["status"] == "refused"] == [
        ("apply", "refused")
    ], new
    assert new[-1]["status"] == "refused" and "gateway" in str(new[-1].get("detail")).lower()
    assert not any(r["step"] in STEPS[2:] for r in new), new
    tl = box.timeline()[n_tl:]
    assert not [r for r in tl if r["tool"] == "chown"]
    subs = [sub for _, _, sub, _ in compose_calls(tl)]
    assert not set(subs) & {"pull", "run", "up"}, f"the refused apply ran compose {subs}"
    assert snapshot(box.run_loom) == before_files, "the refused apply changed run/loom"
    assert (box.var_lib / "release.json").read_bytes() == before_release


# ---------------------------------------------------------------- the manifest


def stand_in_with(extra: bytes) -> dict[str, bytes]:
    return {**STAND_IN, "compose.prod.yml": STAND_IN["compose.prod.yml"] + extra}


@pytest.mark.parametrize("bad", ["compose_sha256", "bundle_sha256", "bundle_compose"])
def test_a_manifest_that_does_not_match_its_files_fails(box: Box, bad: str) -> None:
    """§6.12: both sha256s are checked, and the bundle's compose.prod.yml must be the one
    the manifest names. A failure (exit 1), not a refusal; nothing is pulled or started."""
    if bad == "bundle_compose":
        box.publish(SHA_A, bundle_files=stand_in_with(b"# changed\n"),
                    compose_bytes=STAND_IN["compose.prod.yml"])  # fmt: skip
    else:
        doc = box.publish(SHA_A, bundle_files=STAND_IN)
        key = bad.removesuffix("_sha256")
        doc[key] = {**doc[key], "sha256": hashlib.sha256(b"something else").hexdigest()}
        box.put_object(f"releases/{SHA_A}.json", json.dumps(doc).encode())
    proc = box.deploy("apply", "--release", SHA_A)
    assert_exit(proc, 1)
    recs = box.log_records()
    assert any(r["step"] == "manifest" and r["status"] == "failed" for r in recs), recs
    tl = box.timeline()
    assert not [r for r in tl if r["tool"] == "docker" and r["argv"][:1] == ["login"]]
    subs = [sub for _, _, sub, _ in compose_calls(tl)]
    assert not set(subs) & {"pull", "run", "up"}, subs


# ---------------------------------------------------------------- check_database_url (D10)

GOOD = "postgresql://loom_app:pw@db:5432/loom_platform?sslmode=verify-full&sslrootcert=/etc/loom/rds-ca.pem"


@pytest.mark.parametrize(
    "url",
    [
        GOOD,
        "postgresql://loom_app:pw@db:5432/loom_platform?sslrootcert=/etc/loom/rds-ca.pem&sslmode=verify-full",
        "postgresql://loom_owner:p%40ss@db.example:5432/loom_platform?sslmode=verify-full&sslrootcert=/etc/loom/rds-ca.pem&connect_timeout=5",
    ],
)
def test_check_database_url_accepts_verify_full_with_the_rds_ca(url: str) -> None:
    assert deploy_module().check_database_url(url) is None


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://loom_app:pw@db:5432/loom_platform",
        "postgresql://loom_app:pw@db:5432/loom_platform?sslmode=require&sslrootcert=/etc/loom/rds-ca.pem",
        "postgresql://loom_app:pw@db:5432/loom_platform?sslmode=verify-ca&sslrootcert=/etc/loom/rds-ca.pem",
        "postgresql://loom_app:pw@db:5432/loom_platform?sslmode=verify-full",
        "postgresql://loom_app:pw@db:5432/loom_platform?sslmode=verify-full&sslrootcert=/tmp/ca.pem",
        "postgresql://loom_app:pw@db:5432/loom_platform?sslrootcert=/etc/loom/rds-ca.pem",
        "postgresql://loom_app:sslmode=verify-full@db:5432/loom_platform?sslrootcert=/etc/loom/rds-ca.pem",
    ],
)
def test_check_database_url_refuses_anything_else(url: str) -> None:
    d = deploy_module()
    with pytest.raises(d.Refused):
        d.check_database_url(url)


def test_refused_is_an_exception() -> None:
    d = deploy_module()
    assert isinstance(d.Refused, type) and issubclass(d.Refused, Exception)
    assert not issubclass(d.Refused, (TypeError, KeyError, AttributeError))


# ---------------------------------------------------------------- uids vs Dockerfiles

DOCKERFILES = {
    "gateway": ROOT / "services" / "gateway" / "Dockerfile",
    "billing": ROOT / "services" / "billing" / "Dockerfile",
    "web": ROOT / "apps" / "web" / "Dockerfile",
    "poolmgr": ROOT / "services" / "poolmgr" / "Dockerfile",
    "migrate": ROOT / "services" / "devstack" / "Dockerfile",
    "acceptance": ROOT / "services" / "devstack" / "Dockerfile",
}


def image_uid(dockerfile: Path) -> int:
    """The uid the final stage runs as: its last `USER` (a number, or a name created with
    `useradd --uid N` / `adduser -u N`), else root."""
    text = dockerfile.read_text(encoding="utf-8")
    final = re.split(r"^\s*FROM\s", text, flags=re.M | re.I)[-1]
    users = re.findall(r"^\s*USER\s+(\S+)", final, re.M | re.I)
    if not users:
        return 0
    user = users[-1].split(":")[0]
    if user.isdigit():
        return int(user)
    for line in re.findall(r"^.*\b(?:useradd|adduser)\b.*$", text, re.M):
        m = re.search(r"(?:--uid|-u)[ =](\d+)", line)
        if m and re.search(rf"\s{re.escape(user)}\s*(?:\\|&&|$)", line):
            return int(m.group(1))
    raise AssertionError(f"{dockerfile}: USER {user} has no uid in the file")


@pytest.mark.parametrize("svc", sorted(DOCKERFILES))
def test_the_uid_table_matches_the_dockerfiles(svc: str) -> None:
    assert image_uid(DOCKERFILES[svc]) == UIDS[svc], (
        f"{svc}: its Dockerfile runs as {image_uid(DOCKERFILES[svc])}, §6.12 says {UIDS[svc]}"
    )


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
