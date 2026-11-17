"""M2 acceptance on the running compose stack: the pool manager on its local backend
(docs/phase1-m2-spec.md §9, §13 item 25).

Drives the stack from the outside: requests through the gateway (plain HTTP, as any
client), the manager's admin CLI in its container (`loom-poolmgr stop|warm|reset|
local-capacity`), the pool hashes in Redis and `serving_pods` in Postgres as the
evidence, and `pod_tool.py` to kill a pod's engine inside the poolmgr container.

    docker compose up -d --build --wait
    uv run loom-m2-acceptance --json /tmp/m2.json    # ~20 minutes; exit 0 = every check passed

Checks (one report item each, in the order they run):

cold_wake_ready    a stopped (cold) model: 503 model_unavailable + Retry-After and a wake,
                   then `warming` with a falling Retry-After, then `ready` and a 200
pod_kill_replaced  a pod's engine killed mid-stream: upstream_lost + [DONE], the streamed
                   tokens billed, the pod failed and a new one ready
no_capacity        `loom-poolmgr local-capacity off`: the wake meets no capacity, the pool
                   goes `unavailable` / `no_capacity`, 503 "no GPU capacity" with the
                   backoff's Retry-After; capacity back on, the next retry serves
spend_cap          a synthetic terminated `serving_pods` row puts month-to-date over the
                   cap: the wake is refused before any create, pool `unavailable` /
                   `spend_cap`; the row and any alert rows it caused are deleted after,
                   then `reset`
idle_stop          a ready model with no demand for its idle_stop_s (compose's dev
                   override, `LOOM_POOLMGR_IDLE_STOP_S`, 600 s) is drained to `cold`

The tenant (an approved org with a key and raised limits) is written straight into
Postgres: this run checks the manager, M1's acceptance checks the sign-up path. Ports
and the compose project come from the same variables as `loom-m1-acceptance`.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import itertools
import json
import secrets
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import redis

from loom_devstack import acceptance as m1
from loom_devstack.acceptance import (
    AcceptanceError,
    Db,
    ItemResult,
    Report,
    Stack,
    check,
    ensure_ready,
    live_pods,
    pod_tool,
    pool_of,
    poolmgr,
    ready_replica,
    say,
    sse_events,
    stop_model,
    wait_replaced,
)
from loom_platform.apikeys import generate_api_key

MODEL = "qwen3-8b"
CHECKS = ("cold_wake_ready", "pod_kill_replaced", "no_capacity", "spend_cap", "idle_stop")
TITLES = {
    "cold_wake_ready": "Cold model: 503 + Retry-After, wake, warming, ready",
    "pod_kill_replaced": "Engine killed mid-stream: upstream_lost, billed, pod replaced",
    "no_capacity": "No capacity (local switch): 503 'no GPU capacity', then served",
    "spend_cap": "Create past the monthly cap refused: unavailable / spend_cap",
    "idle_stop": "Ready model idle for idle_stop_s: drained to cold",
}
COMMITTED_SERVING_YAML = Path(__file__).resolve().parents[4] / "config" / "serving.yaml"


class M2:
    def __init__(self, stack: Stack, model: str) -> None:
        self.stack = stack
        self.model = model
        self.db = Db(stack.database_url)
        self.redis = redis.Redis.from_url(stack.redis_url)
        self.key = self._tenant()
        self.http = httpx.Client(timeout=60, trust_env=False)

    def close(self) -> None:
        self.http.close()
        self.db.conn.close()
        self.redis.close()

    def _tenant(self) -> str:
        """An approved org with $5 of credit, raised limits and one API key."""
        tag = secrets.token_hex(4)
        email = f"m2-{tag}@m2.test"
        user = self.db.one(
            "INSERT INTO users (name, email, role) VALUES (%s, %s, 'admin') RETURNING id",
            f"m2-{tag}",
            email,
        )
        org = self.db.one(
            "INSERT INTO orgs (name, status) VALUES (%s, 'pending_approval') RETURNING id",
            f"m2 acceptance {tag}",
        )
        self.db.conn.execute(
            "INSERT INTO org_members (org_id, user_id, role) VALUES (%s, %s, 'owner')",
            (org, user),
        )
        self.db.one("SELECT loom_approve_org(%s, %s, %s)", org, user, 5_000_000)
        key = generate_api_key()
        self.db.conn.execute(
            "INSERT INTO api_keys (org_id, created_by, name, prefix, key_hash)"
            " VALUES (%s, %s, 'm2-acceptance', %s, %s)",
            (org, user, key.prefix, key.key_hash),
        )
        self.db.raise_limits(str(org))
        return key.plaintext

    # -- helpers

    def post(self, **body: Any) -> httpx.Response:
        return self.http.post(
            f"{self.stack.gateway}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
                **body,
            },
        )

    def wake_set(self) -> bool:
        return bool(self.redis.exists(f"loom:pool:{self.model}:wake"))

    def wait_pool(
        self, pred: Callable[[dict[str, Any]], bool], timeout_s: float, what: str
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            pool = pool_of(self.redis, self.model)
            if pred(pool):
                return pool
            time.sleep(0.5)
        raise AcceptanceError(f"{what} within {timeout_s} s: {pool_of(self.redis, self.model)}")

    def unavailable_503(self, res: ItemResult, reason: str) -> None:
        """The first 503 (cold) wakes the model; the pool then says `unavailable` /
        `reason`, and the next 503 carries the pool's Retry-After."""
        r = self.post()
        check(r.status_code == 503, f"first request: {r.status_code} {r.text[:200]}")
        check(r.json()["error"]["code"] == "model_unavailable", f"first 503: {r.text[:200]}")
        check(self.wake_set(), "the 503 set no wake key")
        pool = self.wait_pool(
            lambda p: p.get("state") == "unavailable" and p.get("reason") == reason,
            60,
            f"pool not unavailable / {reason}",
        )
        r = self.post()
        check(r.status_code == 503, f"{reason}: {r.status_code} {r.text[:200]}")
        err = r.json()["error"]
        check(err["code"] == "model_unavailable", f"{reason}: code {err['code']}")
        check("no GPU capacity" in err["message"], f"{reason}: message {err['message']!r}")
        check(
            r.headers.get("retry-after") == str(pool["retry_after_s"]),
            f"{reason}: Retry-After {r.headers.get('retry-after')} != pool {pool['retry_after_s']}",
        )
        res.numbers.update(
            pool_state=pool["state"],
            pool_reason=pool.get("reason"),
            message=err["message"],
            retry_after_s=int(r.headers["retry-after"]),
        )

    # -- the checks

    def cold_wake_ready(self, res: ItemResult) -> None:
        res.numbers["stopped_after_s"] = round(
            stop_model(self.stack, self.redis, self.db, self.model), 1
        )
        cold = pool_of(self.redis, self.model)
        t0 = time.monotonic()
        r = self.post()
        check(r.status_code == 503, f"cold: {r.status_code} {r.text[:200]}")
        err = r.json()["error"]
        check(err["code"] == "model_unavailable", f"cold: code {err['code']}")
        check(
            r.headers.get("retry-after") == str(cold.get("retry_after_s")),
            f"cold Retry-After {r.headers.get('retry-after')} != pool {cold.get('retry_after_s')}",
        )
        check(self.wake_set(), "the cold 503 set no wake key")
        warming: list[int] = []
        states: list[str] = []
        while time.monotonic() - t0 < 180:
            pool = pool_of(self.redis, self.model)
            state = pool.get("state", "absent")
            if not states or states[-1] != state:
                states.append(state)
            if state == "warming":
                ra = int(pool["retry_after_s"])
                if not warming or warming[-1] != ra:
                    warming.append(ra)
            if state == "ready" and pool["replicas"]:
                break
            time.sleep(0.5)
        check(states[-1] == "ready", f"never ready: states {states}")
        ready_s = time.monotonic() - t0
        check(warming, f"never seen warming: states {states}")
        check(
            all(a >= b for a, b in itertools.pairwise(warming)),
            f"warming Retry-After rose: {warming}",
        )
        r = self.post()
        check(r.status_code == 200, f"ready: {r.status_code} {r.text[:200]}")
        pod = ready_replica(self.redis, self.model)
        assert pod is not None
        stages = self.db.one("SELECT stages FROM serving_pods WHERE pod_id = %s", pod["id"])
        check(stages, f"no stages recorded for {pod['id']}")
        res.numbers.update(
            first_status=503,
            message=err["message"],
            cold_retry_after_s=int(cold["retry_after_s"]),
            states=states,
            warming_retry_after_s=warming[:10],
            ready_after_s=round(ready_s, 1),
            pod=pod["id"],
            stages_s={k: round(float(v), 1) for k, v in dict(stages).items()},
        )

    def pod_kill_replaced(self, res: ItemResult) -> None:
        ensure_ready(self.stack, self.redis, self.model)
        rep = ready_replica(self.redis, self.model)
        assert rep is not None
        pod = str(rep["id"])
        body = {
            "messages": [{"role": "user", "content": f"Count slowly {secrets.token_hex(4)}."}],
            "max_tokens": 400,
            "min_tokens": 200,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
        events: list[dict[str, Any] | str] = []
        killed = False
        with self.http.stream(
            "POST",
            f"{self.stack.gateway}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, **body},
        ) as r:
            check(r.status_code == 200, f"stream: {r.status_code}")
            rid = r.headers["x-loom-request-id"]
            content = 0
            for ev in sse_events(r.iter_lines()):
                events.append(ev)
                if isinstance(ev, dict) and ev.get("choices"):
                    if ev["choices"][0].get("delta", {}).get("content"):
                        content += 1
                    if content == 10 and not killed:
                        pod_tool(self.stack, "kill", pod)
                        killed = True
        check(killed, f"the stream ended after {content} chunks, before the kill")
        check(events and events[-1] == "[DONE]", "the stream did not end with [DONE]")
        errs = [e for e in events if isinstance(e, dict) and "error" in e]
        check(
            bool(errs) and errs[-1]["error"]["code"] == "upstream_lost",
            f"no upstream_lost error chunk: {errs}",
        )
        usages = [e["usage"] for e in events if isinstance(e, dict) and e.get("usage")]
        check(usages, "no per-chunk usage")
        last = usages[-1]
        row = self.db.usage([rid])[rid]
        check(row["status"] == "aborted", f"status {row['status']}")
        got = (row["prompt_tokens"], row["completion_tokens"])
        want = (last["prompt_tokens"], last["completion_tokens"])
        check(got == want, f"billed {got} != the last usage the client saw {want}")
        check(10 <= last["completion_tokens"] < 200, f"not cut mid-way: {last}")

        replaced = wait_replaced(self.stack, self.redis, self.model, pod)
        new = ready_replica(self.redis, self.model)
        assert new is not None
        old = self.db.rows(
            "SELECT state, failure, terminated_by FROM serving_pods WHERE pod_id = %s", pod
        )
        check(old and old[0][1], f"the killed pod's row has no failure: {old}")
        check(old[0][0] in ("failed", "terminated"), f"the killed pod's row: {old}")
        check(live_pods(self.db, self.model) == 1, "more than one live pod after the replacement")
        r2 = self.post()
        check(r2.status_code == 200, f"replacement: {r2.status_code} {r2.text[:200]}")
        # The kill is deliberate: keep it out of the crash-loop count for later runs.
        poolmgr(self.stack, "reset", self.model)
        res.numbers.update(
            killed_pod=pod,
            client_saw_error=errs[-1]["error"]["code"],
            billed_completion_tokens=row["completion_tokens"],
            billed_prompt_tokens=row["prompt_tokens"],
            cost_micros=row["cost_micros"],
            old_row={"state": old[0][0], "failure": old[0][1], "terminated_by": old[0][2]},
            replacement=new["id"],
            replaced_after_s=round(replaced, 1),
        )

    def no_capacity(self, res: ItemResult) -> None:
        stop_model(self.stack, self.redis, self.db, self.model)
        poolmgr(self.stack, "local-capacity", "off")
        try:
            self.unavailable_503(res, "no_capacity")
            check(live_pods(self.db, self.model) == 0, "a pod is live without capacity")
        finally:
            poolmgr(self.stack, "local-capacity", "on")
        # Capacity back: the manager retries on its backoff while the wake lives.
        t0 = time.monotonic()
        self.wait_pool(
            lambda p: p.get("state") == "ready" and bool(p["replicas"]),
            240,
            "not served again after capacity came back",
        )
        r = self.post()
        check(r.status_code == 200, f"after capacity: {r.status_code} {r.text[:200]}")
        res.numbers["served_again_after_s"] = round(time.monotonic() - t0, 1)
        res.numbers["no_capacity_rows"] = int(
            self.db.one(
                "SELECT count(*) FROM serving_pods WHERE model_id = %s AND failure = 'no_capacity'"
                " AND created_at > now() - interval '10 minutes'",
                self.model,
            )
        )

    def spend_cap(self, res: ItemResult) -> None:
        stop_model(self.stack, self.redis, self.db, self.model)
        alerts_before = set(self.db.rows("SELECT month, fraction FROM serving_spend_alerts"))
        rows_before = int(self.db.one("SELECT count(*) FROM serving_pods"))
        row_id = uuid.uuid4()
        now = dt.datetime.now(dt.UTC)
        # $1M/h for a minute: ~$16,667 this month, far past any cap.
        self.db.conn.execute(
            "INSERT INTO serving_pods (id, model_id, pod_name, state, cost_per_hr_micros,"
            " failure, terminated_by, created_at, terminated_at)"
            " VALUES (%s, %s, %s, 'terminated', %s, %s, 'admin', %s, %s)",
            (
                row_id,
                self.model,
                f"m2-acceptance-spend-{secrets.token_hex(3)}",
                1_000_000 * 1_000_000,
                "synthetic: loom-m2-acceptance spend_cap",
                now - dt.timedelta(seconds=60),
                now,
            ),
        )
        try:
            self.unavailable_503(res, "spend_cap")
            created = int(self.db.one("SELECT count(*) FROM serving_pods")) - rows_before - 1
            check(created == 0, f"{created} serving_pods row(s) written past the cap")
            check(live_pods(self.db, self.model) == 0, "a pod is live past the cap")
        finally:
            self.db.conn.execute("DELETE FROM serving_pods WHERE id = %s", (row_id,))
            poolmgr(self.stack, "reset", self.model)
        try:
            res.numbers["served_again_after_reset_s"] = round(
                ensure_ready(self.stack, self.redis, self.model), 1
            )
        finally:
            # After the ticks that served it again, so no tick still holding the old
            # month-to-date writes an alert after this.
            caused = [
                a
                for a in self.db.rows("SELECT month, fraction FROM serving_spend_alerts")
                if a not in alerts_before
            ]
            for month, fraction in caused:
                self.db.conn.execute(
                    "DELETE FROM serving_spend_alerts WHERE month = %s AND fraction = %s",
                    (month, fraction),
                )
            res.numbers["alerts_caused_and_deleted"] = [float(f) for _, f in caused]

    def idle_stop(self, res: ItemResult) -> None:
        idle_s = self.idle_stop_s()
        check(0 < idle_s <= 900, f"idle_stop_s {idle_s}: set LOOM_POOLMGR_IDLE_STOP_S <= 900")
        ensure_ready(self.stack, self.redis, self.model)
        # No demand from here on but the one request below: drop the wake the warm-up
        # left (it counts as demand while it lives, up to 600 s).
        self.redis.delete(f"loom:pool:{self.model}:wake")
        pod = ready_replica(self.redis, self.model)
        assert pod is not None
        r = self.post()
        check(r.status_code == 200, f"last request: {r.status_code}")
        t0 = time.monotonic()
        pool = self.wait_pool(
            lambda p: p.get("state") != "ready", idle_s + 120, "not drained after idle_stop_s"
        )
        out_after = time.monotonic() - t0
        check(pool.get("state") == "cold", f"after the idle stop: {pool}")
        check(
            out_after >= idle_s - 2,
            f"out of routing after {out_after:.0f} s, before idle_stop_s {idle_s}",
        )
        deadline = time.monotonic() + 120
        row: list[tuple[Any, ...]] = []
        while time.monotonic() < deadline:
            row = self.db.rows(
                "SELECT state, terminated_by, failure, drain_started_at IS NOT NULL"
                " FROM serving_pods WHERE pod_id = %s",
                pod["id"],
            )
            if row and row[0][0] == "terminated":
                break
            time.sleep(1)
        check(row and row[0][0] == "terminated", f"the idle pod was not terminated: {row}")
        check(row[0][1] == "manager" and not row[0][2], f"not a clean idle stop: {row}")
        check(row[0][3], "terminated without a drain")
        res.numbers.update(
            idle_stop_s=idle_s,
            pod=pod["id"],
            out_of_routing_after_s=round(out_after, 1),
            terminated_after_s=round(time.monotonic() - t0, 1),
            final_pool_state=pool.get("state"),
        )

    def idle_stop_s(self) -> int:
        """The manager's effective idle_stop_s: compose's dev override, else serving.yaml."""
        with contextlib.suppress(AcceptanceError):
            value = poolmgr_env(self.stack, "LOOM_POOLMGR_IDLE_STOP_S")
            if value:
                return int(value)
        from loom_platform.serving import load_serving

        config = load_serving(COMMITTED_SERVING_YAML, check_registry=False)
        return int(config.models[self.model].idle_stop_s)


def poolmgr_env(stack: Stack, name: str) -> str:
    out = m1.docker("exec", m1.container(stack, m1.POOLMGR), "printenv", name)
    return out.strip()


def run(stack: Stack, model: str, report: Report, only: tuple[str, ...] = CHECKS) -> None:
    acc = M2(stack, model)
    try:
        for name in CHECKS:
            if name not in only:
                continue
            res = ItemResult(name, TITLES[name])
            say(f"\n== {name}: {res.title}")
            t0 = time.monotonic()
            try:
                getattr(acc, name)(res)
            except Exception as e:  # recorded, and the run goes on
                res.status = "FAIL"
                res.notes.append(f"{type(e).__name__}: {e}")
                with contextlib.suppress(Exception):
                    poolmgr(stack, "local-capacity", "on")
            res.seconds = round(time.monotonic() - t0, 1)
            report.items.append(res)
            say(f"   {res.status} ({res.seconds} s) {json.dumps(res.numbers, default=str)}")
            for n in res.notes:
                say(f"   note: {n}")
    finally:
        acc.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="loom-m2-acceptance", description=__doc__.splitlines()[0])
    p.add_argument("--gateway", help="gateway base URL including /v1")
    p.add_argument("--database-url", dest="database_url")
    p.add_argument("--redis-url", dest="redis_url")
    p.add_argument("--project", help="compose project name (default $COMPOSE_PROJECT_NAME or loom)")
    p.add_argument("--model", default=MODEL, help=f"the model to drive (default {MODEL})")
    p.add_argument("--only", default=",".join(CHECKS), help="comma-separated checks (default: all)")
    p.add_argument("--json", type=Path, help="write the results here")
    args = p.parse_args(argv)
    stack = m1.stack_from_env(args)
    only = tuple(c.strip() for c in args.only.split(",") if c.strip())
    unknown = set(only) - set(CHECKS)
    if unknown:
        p.error(f"unknown checks {sorted(unknown)}")
    report = Report()
    say(f"M2 acceptance against {stack.gateway} (project {stack.project}, model {args.model})")
    run(stack, args.model, report, only)
    say("\n== summary")
    for i in report.items:
        say(f"{i.item:18} {i.status:4}  {i.title}")
    if args.json:
        args.json.write_text(json.dumps(report.to_json(), indent=2, default=str))
    sys.exit(0 if report.passed and report.items else 1)


if __name__ == "__main__":
    main()
