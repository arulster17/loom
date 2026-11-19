"""M3 acceptance on the box (docs/phase1-m3-spec.md §10, B10-B19).

Runs in a one-shot container on the production compose network (`loom-deploy
acceptance` starts it through SSM): the web app, the gateway and the Stripe webhook are
reached **through the public URLs** (real DNS, real TLS), Postgres, Redis and the
containers from inside, as `loom-m1-acceptance` does (its helpers are reused).

    loom-m3-acceptance --domain <domain> --json /var/lib/loom/m3-acceptance.json
    loom-m3-acceptance --external --domain <domain> --brand-name <name>   # operator's machine

Sign-in: test users are signed in by inserting an Auth.js database session and sending
its cookie; the admin is granted with contracts §4's SQL. The acceptance's users
(`<who>-<run>@m3-acceptance.test`) and their orgs are left in place, as M1's are.

Items (box mode runs all but m3-h; `--external` runs only m3-h):

m3-0  M1's flow through https://api.<domain>/v1: sign up -> pending -> approve -> $5 -> key
      -> SDK (both models, stream and not, chat and completions) -> playground -> usage
      rows and balance drop match the engines' counters
m3-b  a revoked key (and a suspended org) refused within 5 s, through the public URL
m3-d  gateway p50 added overhead < 10 ms (M1 item d's probe from the poolmgr container);
      the Caddy + TLS hop against the public URL is measured and reported, not gated
m3-g  Stripe test mode: a checkout as a test user, **the operator pays** (4242...); one
      purchase keyed stripe:<event id>, the Redis credit, the dashboard balance; a re-post
      with a fresh signature -> 200 duplicate; an old `t` -> 400; **the operator resends**
      -> duplicate; **the operator refunds** part, then the rest -> two refunds summing to
      the purchase
m3-s  missed delivery: billing stopped, **the operator pays** again (Caddy answers 502),
      H1/H3 alarm emails (checked by the operator), billing started, **the operator
      resends** -> exactly one purchase; a second resend -> duplicate
m3-h  exposure (external): only 80/443 answer; api. serves /v1/* only; hooks. only POST
      /stripe/webhook and GET /stripe/health; /internal, /health, /metrics unreachable;
      80 redirects; HSTS; a valid certificate for the three names; the sign-in page's
      title and heading show the product name
m3-i  no prompt, completion or API key in any container's logs or the CloudWatch log group
m3-p  P4: the deploy log shows the gateway healthy before the pool manager was updated;
      `loom-deploy apply --only poolmgr` with a newer pool manager is refused
m3-x  containers cannot reach the instance metadata service (IMDS hop limit 1)

Manual steps wait (polling) up to `--manual-timeout-s`; the JSON report says which steps
were manual. m3-a (alarm routes) is the runbook's, by hand.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import ssl
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from loom_devstack import acceptance as m1
from loom_devstack.acceptance import (
    AcceptanceError,
    Browser,
    ItemResult,
    Report,
    Stack,
    check,
    docker,
    page_errors,
    say,
    visible,
)

BOX_ITEMS = ("m3-0", "m3-b", "m3-d", "m3-g", "m3-s", "m3-i", "m3-p", "m3-x")
EXTERNAL_ITEMS = ("m3-h",)
ITEMS = BOX_ITEMS + EXTERNAL_ITEMS
TITLES = {
    "m3-0": "M1's flow through the public URLs, usage and balance match the engines",
    "m3-b": "Revoked key and suspended org refused within 5 s (public URL)",
    "m3-d": "Gateway p50 added overhead < 10 ms; Caddy + TLS hop reported",
    "m3-g": "Stripe test mode: pay, one purchase, duplicate, stale t, resend, refunds",
    "m3-s": "Missed delivery while billing is stopped, then resent: one purchase",
    "m3-h": "Exposure from the internet: routes, ports, redirect, HSTS, certificate, name",
    "m3-i": "No prompt, completion or key in container logs or CloudWatch",
    "m3-p": "P4: gateway healthy before poolmgr; --only poolmgr newer is refused",
    "m3-x": "Containers cannot reach the instance metadata service",
}
MANUAL = {
    "m3-g": [
        "operator pays with 4242 4242 4242 4242",
        "operator resends",
        "operator refunds twice",
    ],
    "m3-s": [
        "operator pays",
        "operator checks the H1 and H3 alarm emails",
        "operator resends twice",
    ],
}
CLOSED_PORTS = (22, 3000, 3001, 5432, 6379, 8080, 8090, 9090, 9100, 9101, 9102)
SESSION_COOKIES = ("__Secure-authjs.session-token", "authjs.session-token")


def public_stack(args: argparse.Namespace) -> Stack:
    d = args.domain
    return Stack(
        web=f"https://{d}",
        gateway=f"https://api.{d}/v1",
        mailpit="",
        database_url=args.database_url or os.environ.get("LOOM_ACCEPT_DATABASE_URL", ""),
        redis_url=args.redis_url or os.environ.get("LOOM_ACCEPT_REDIS_URL", ""),
        project=args.project,
        admin_email=args.admin_email or "",
        gateway_metrics=args.gateway_metrics,
        billing=f"https://hooks.{d}",
        stub_secret_file=args.webhook_secret_file or "",
    )


def sign(secret: str, body: bytes, ts: int) -> str:
    v1 = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={v1}"


def metric_value(url: str, name: str, labels: dict[str, str] | None = None) -> float:
    text = httpx.get(url, timeout=10, trust_env=False).text
    want = labels or {}
    total = 0.0
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        m = re.match(rf"{re.escape(name)}(\{{[^}}]*\}})? (\S+)", line)
        if not m:
            continue
        got = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1) or ""))
        if all(got.get(k) == v for k, v in want.items()):
            total += float(m.group(2))
    return total


def wait_for(what: str, cond: Any, timeout_s: float, every_s: float = 3) -> float:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if cond():
            return time.monotonic() - t0
        time.sleep(every_s)
    raise AcceptanceError(f"{what}: not within {timeout_s:.0f} s")


class M3Acceptance(m1.Acceptance):
    def __init__(self, stack: Stack, report: Report, args: argparse.Namespace) -> None:
        super().__init__(stack, report)
        self.args = args

    def email(self, who: str) -> str:
        return f"{who}-{self.run_id}@m3-acceptance.test"

    # -- sign-in: an Auth.js database session, no email (§10, D6)

    def session_browser(self, email: str) -> Browser:
        uid = self.db.one(
            'INSERT INTO users (email, "emailVerified") VALUES (%s, now())'
            " ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email RETURNING id::text",
            email,
        )
        token = secrets.token_hex(32)
        self.db.conn.execute(
            'INSERT INTO sessions ("userId", expires, "sessionToken")'
            " VALUES (%s, now() + interval '1 day', %s)",
            (uid, token),
        )
        b = Browser(self.stack.web, email)
        for name in SESSION_COOKIES:
            b.http.cookies.set(name, token, domain=self.args.domain, path="/")
        return b

    def signup_and_approve(self, who: str) -> tuple[Browser, str]:
        email = self.email(who)
        b = self.session_browser(email)
        self.browsers[who] = b
        r = b.get("/dashboard")
        check(r.status_code == 200, f"{who}: /dashboard {r.status_code}")
        check("awaiting approval" in r.text, f"{who}: a new signup should see the pending page")
        org = self.db.org_of(email)
        check(
            self.db.one("SELECT status FROM orgs WHERE id = %s", org) == "pending_approval",
            "new org is not pending_approval",
        )
        self.approve(org)
        return b, org

    def admin_browser(self) -> Browser:
        if self.admin is None:
            email = self.stack.admin_email or self.email("admin")
            b = self.session_browser(email)
            b.get("/dashboard")  # the users row and its org exist after the first visit
            self.db.conn.execute(
                "WITH granted AS (UPDATE users SET role = 'admin'"
                " WHERE email = %s AND role = 'user' RETURNING id)"
                " INSERT INTO admin_audit_log (actor_user_id, action, target_type, target_id,"
                " details) SELECT id, 'user.grant_admin', 'user', id::text,"
                ' \'{"via": "sql"}\' FROM granted',
                (email,),
            )
            r = b.get("/admin")
            check(r.status_code == 200 and "Approve signups" in r.text, f"{email} is not an admin")
            self.admin = b
        return self.admin

    # -- m3-0, m3-b: M1's items 0 and b through the public URLs

    def item_m3_0(self, res: ItemResult) -> None:
        self.flow(res)

    def item_m3_b(self, res: ItemResult) -> None:
        self.item_b(res)

    # -- m3-d

    def item_m3_d(self, res: ItemResult) -> None:
        self.item_d(res)
        key = self.load_keys()[0]
        hdr = {"Authorization": f"Bearer {key}"}

        def timings(url: str, n: int = 200) -> list[float]:
            out = []
            with httpx.Client(timeout=30, trust_env=False) as http:
                http.get(url, headers=hdr)
                for _ in range(n):
                    t0 = time.perf_counter()
                    r = http.get(url, headers=hdr)
                    out.append((time.perf_counter() - t0) * 1000)
                    check(r.status_code == 200, f"GET {url}: {r.status_code}")
            return out

        public = timings(f"{self.stack.gateway}/models")
        direct = timings(f"{self.args.gateway_internal}/v1/models")
        res.numbers["caddy_tls_hop_p50_ms"] = round(
            statistics.median(public) - statistics.median(direct), 3
        )
        res.numbers["public_models_p50_ms"] = round(statistics.median(public), 3)
        res.notes.append("the Caddy + TLS hop is reported, not gated (§10)")

    # -- m3-g

    def checkout(self, b: Browser, amount: str) -> str:
        r = b.submit(
            "/dashboard/billing",
            lambda f: f.has("amount") and not f.has("threshold"),
            {"amount": amount},
        )
        check(r.status_code in (302, 303), f"add credit: {r.status_code} {page_errors(r.text)}")
        url = r.headers["location"]
        check(url.startswith("https://checkout.stripe.com/"), f"checkout redirect {url!r}")
        return url

    def purchases(self, org: str) -> list[tuple[Any, ...]]:
        return self.db.rows(
            "SELECT id::text, idempotency_key, external_ref FROM ledger_transactions"
            " WHERE org_id = %s AND kind = 'purchase' ORDER BY created_at",
            org,
        )

    def refunds(self, org: str) -> int:
        return int(
            self.db.one(
                "SELECT count(*) FROM ledger_transactions WHERE org_id = %s AND kind = 'refund'",
                org,
            )
        )

    def duplicates(self) -> float:
        return metric_value(
            self.args.billing_metrics, "loom_billing_stripe_webhooks_total", {"result": "duplicate"}
        )

    def post_event(self, event_id: str, ts: int) -> httpx.Response:
        payload = self.db.one("SELECT payload::text FROM stripe_events WHERE id = %s", event_id)
        check(payload, f"stripe event {event_id} not stored")
        secret = Path(self.args.webhook_secret_file).read_text().strip()
        body = str(payload).encode()
        return httpx.post(
            f"{self.stack.billing}/stripe/webhook",
            content=body,
            headers={
                "Stripe-Signature": sign(secret, body, ts),
                "Content-Type": "application/json",
            },
            timeout=30,
        )

    def item_m3_g(self, res: ItemResult) -> None:
        b, org = self.signup_and_approve("payer")
        credit0, _, _ = self.db.balance(org)
        url = self.checkout(b, "10")
        say(f"   OPERATOR: pay $10 with 4242 4242 4242 4242 at\n   {url}")
        res.numbers["paid_after_s"] = round(
            wait_for("the purchase", lambda: self.purchases(org), self.args.manual_timeout_s), 1
        )
        time.sleep(5)
        bought = self.purchases(org)
        check(len(bought) == 1, f"purchases: {bought}")
        key = str(bought[0][1])
        check(key.startswith("stripe:evt_"), f"purchase keyed {key!r}, not stripe:<event id>")
        event_id = key.removeprefix("stripe:")
        credit1, usage1, _ = self.db.balance(org)
        check(credit1 - credit0 == 10_000_000, f"credit rose by {credit1 - credit0}")
        check(
            int(self.redis.get(m1.org_key(org, "credit")) or -1) == credit1,
            "Redis credit != Postgres",
        )
        bill = visible(b.get("/dashboard/billing").text)
        check(m1.usd_cents(credit1 - usage1) in bill, "the dashboard does not show the new balance")

        dup0 = self.duplicates()
        again = self.post_event(event_id, int(time.time()))
        check(again.status_code == 200, f"re-post: {again.status_code}")
        check(again.json().get("result") == "duplicate", f"re-post: {again.text[:200]}")
        stale = self.post_event(event_id, int(time.time()) - 3600)
        check(stale.status_code == 400, f"old t: {stale.status_code}")
        check(self.purchases(org) == bought, "a re-post changed the ledger")

        say(f"   OPERATOR: resend event {event_id} from the Stripe dashboard")
        wait_for(
            "the resent event",
            lambda: self.duplicates() >= dup0 + 2,
            self.args.manual_timeout_s,
        )
        check(self.purchases(org) == bought, "a resend changed the ledger")

        say("   OPERATOR: refund part of the payment in the Stripe dashboard, then the rest")
        wait_for(
            "two refunds summing to the purchase",
            lambda: credit1 - self.db.balance(org)[0] >= 10_000_000,
            self.args.manual_timeout_s,
        )
        time.sleep(5)
        n = self.refunds(org)
        check(n == 2, f"{n} refund transactions, not 2")
        credit2, _, _ = self.db.balance(org)
        check(credit1 - credit2 == 10_000_000, f"balance down by {credit1 - credit2}")
        res.numbers.update(event_id=event_id, purchase_micros=10_000_000, refunds=n)
        res.notes += MANUAL["m3-g"]

    # -- m3-s

    def item_m3_s(self, res: ItemResult) -> None:
        b, org = self.signup_and_approve("missed")
        billing = m1.container(self.stack, "billing")
        url = self.checkout(b, "5")
        session = url.rstrip("/").rsplit("/", 1)[-1].split("#")[0]
        docker("stop", billing)
        try:
            health = httpx.get(f"{self.stack.billing}/stripe/health", timeout=15)
            check(
                health.status_code == 502,
                f"hooks. health with billing stopped: {health.status_code}",
            )
            say(f"   OPERATOR: pay $5 at\n   {url}\n   then confirm the H1 and H3 alarm emails")
            time.sleep(self.args.alarm_wait_s)
        finally:
            docker("start", billing)
        res.numbers["billing_healthy_after_s"] = round(m1.wait_healthy(billing, 120), 1)
        check(not self.purchases(org), "a purchase arrived while billing was stopped")
        say(f"   OPERATOR: resend the event of checkout {session} from the Stripe dashboard")
        wait_for("the resent purchase", lambda: self.purchases(org), self.args.manual_timeout_s)
        time.sleep(5)
        bought = self.purchases(org)
        check(len(bought) == 1, f"purchases after the resend: {bought}")
        dup0 = self.duplicates()
        say("   OPERATOR: resend the same event once more")
        wait_for("the second resend", lambda: self.duplicates() > dup0, self.args.manual_timeout_s)
        check(self.purchases(org) == bought, "the second resend changed the ledger")
        res.numbers.update(purchase=bought[0][1], session=session)
        res.notes += MANUAL["m3-s"]

    # -- m3-i

    def item_m3_i(self, res: ItemResult) -> None:
        self.item_i(res)
        group = self.args.log_group
        try:
            hits = []
            for c in sorted(self.report.canaries):
                out = subprocess.run(
                    [
                        "aws",
                        "logs",
                        "filter-log-events",
                        "--log-group-name",
                        group,
                        "--filter-pattern",
                        json.dumps(c),
                        "--max-items",
                        "1",
                        "--output",
                        "json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                check(out.returncode == 0, f"aws logs: {out.stderr[-300:]}")
                if json.loads(out.stdout or "{}").get("events"):
                    hits.append(c[:24])
        except (OSError, AcceptanceError) as e:
            raise AcceptanceError(
                f"CloudWatch {group} not scanned ({e}); run m3-i where the aws CLI has the"
                " instance role (loom-deploy acceptance)"
            ) from e
        res.numbers["cloudwatch_hits"] = hits
        check(not hits, f"canaries in CloudWatch {group}: {hits}")

    # -- m3-p

    def item_m3_p(self, res: ItemResult) -> None:
        log = Path(self.args.deploy_log)
        check(log.is_file(), f"no deploy log at {log} (--deploy-log)")
        lines = log.read_text().splitlines()
        gw, pm, gh = p4_log_order(lines, self.args.release)
        res.numbers["deploy_log_lines"] = {
            "gateway_update": gw + 1,
            "gateway_healthy": gh + 1,
            "poolmgr_update": pm + 1,
        }
        if not self.args.p4_release:
            raise AcceptanceError("--p4-release (a release newer than the gateway's) not given")
        poolmgr = m1.container(self.stack, "poolmgr")
        started = docker("inspect", "-f", "{{.State.StartedAt}}", poolmgr).strip()
        cmd = [*self.args.deploy_cmd.split(), "apply", "--only", "poolmgr"]
        r = subprocess.run(
            [*cmd, "--release", self.args.p4_release],
            capture_output=True,
            text=True,
            timeout=600,
        )
        check(r.returncode != 0, "apply --only poolmgr with a newer release was not refused")
        check("gateway" in (r.stdout + r.stderr).lower(), "the refusal does not name the gateway")
        check(
            docker("inspect", "-f", "{{.State.StartedAt}}", poolmgr).strip() == started,
            "poolmgr was restarted by the refused apply",
        )
        res.numbers["only_poolmgr_exit"] = r.returncode

    # -- m3-x

    def item_m3_x(self, res: ItemResult) -> None:
        probe = (
            "import urllib.request\n"
            "try:\n"
            "    req = urllib.request.Request('http://169.254.169.254/latest/api/token',"
            " method='PUT', headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'})\n"
            "    t = urllib.request.urlopen(req, timeout=3).read()\n"
            "    print('TOKEN' if t else 'EMPTY')\n"
            "except Exception as e:\n"
            "    print('BLOCKED', type(e).__name__)\n"
        )
        here = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=30
        )
        results = {"acceptance": here.stdout.strip()}
        for svc in ("gateway", "billing", "poolmgr"):
            results[svc] = docker(
                "exec", "-i", m1.container(self.stack, svc), "python", "-", input_text=probe
            ).strip()
        res.numbers["imds"] = results
        reached = {k: v for k, v in results.items() if not v.startswith("BLOCKED")}
        check(not reached, f"containers reached the instance metadata service: {reached}")

    # -- m3-h (external)

    def item_m3_h(self, res: ItemResult) -> None:
        d = self.args.domain
        hosts = (d, f"api.{d}", f"hooks.{d}")
        nums = res.numbers
        open_ports: dict[str, list[int]] = {}
        for h in hosts:
            for p in CLOSED_PORTS:
                with contextlib.suppress(OSError), socket.create_connection((h, p), timeout=3):
                    open_ports.setdefault(h, []).append(p)
        check(not open_ports, f"ports other than 80/443 answer: {open_ports}")
        for h in hosts:
            ctx = ssl.create_default_context()  # verifies the chain and the host name
            with (
                socket.create_connection((h, 443), timeout=10) as raw,
                ctx.wrap_socket(raw, server_hostname=h) as tls,
            ):
                nums[f"cert_{h}"] = tls.version()
        with httpx.Client(timeout=20, trust_env=False, follow_redirects=False) as http:
            for h in hosts:
                r = http.get(f"http://{h}/")
                check(r.status_code in (301, 302, 307, 308), f"http://{h}/: {r.status_code}")
                check(r.headers.get("location", "").startswith(f"https://{h}"), f"{h} redirect")
            expect = {
                f"https://api.{d}/v1/models": 401,
                f"https://api.{d}/health": 404,
                f"https://api.{d}/metrics": 404,
                f"https://api.{d}/internal/checkout-sessions": 404,
                f"https://api.{d}/": 404,
                f"https://hooks.{d}/health": 404,
                f"https://hooks.{d}/metrics": 404,
                f"https://hooks.{d}/internal/checkout-sessions": 404,
                f"https://hooks.{d}/stripe/webhook": 404,  # GET: only POST is routed
                f"https://hooks.{d}/": 404,
                f"https://{d}/internal/checkout-sessions": 404,
                f"https://{d}/metrics": 404,
            }
            for url, status in expect.items():
                r = http.get(url)
                check(r.status_code == status, f"GET {url}: {r.status_code}, wanted {status}")
                hsts = r.headers.get("strict-transport-security", "")
                check("max-age=31536000" in hsts, f"{url}: HSTS {hsts!r}")
            health = http.get(f"https://hooks.{d}/stripe/health")
            check(health.status_code == 200 and "stripe_webhook" in health.text, "hooks. health")
            post = http.post(f"https://hooks.{d}/stripe/webhook", content=b"{}")
            check(post.status_code == 400, f"unsigned POST /stripe/webhook: {post.status_code}")
            page = http.get(f"https://{d}/signin")
            check(page.status_code == 200, f"/signin: {page.status_code}")
            name = self.args.brand_name
            check(name, "--brand-name (or LOOM_BRAND_NAME) not given")
            title = re.search(r"<title>(.*?)</title>", page.text, re.S)
            heading = re.search(r"<h1[^>]*>(.*?)</h1>", page.text, re.S)
            check(title and name in visible(title.group(1)), f"title lacks {name!r}")
            check(heading and name in visible(heading.group(1)), f"heading lacks {name!r}")
        nums["routes_checked"] = len(expect) + 2


_HEALTHY = re.compile(r"\bhealthy\b", re.I)
_NOT_HEALTHY = re.compile(r"\b(unhealthy|not|wait\w*|until)\b", re.I)
_UPDATE = re.compile(r"\b(up|updat\w*|start\w*|recreat\w*)\b", re.I)


def p4_log_order(lines: list[str], release: str | None) -> tuple[int, int, int]:
    """(gateway update, poolmgr update, gateway healthy) line indexes of the release's
    apply, checking P4: the pool manager is updated only after a record that the gateway
    the same apply updated is healthy.

    The log's format is not fixed by the spec (an assumption: one line per step naming the
    service, as `loom-deploy` prints them). The release's apply starts at the last line
    naming the release with `apply` (else its first mention); a plan line (naming `plan`,
    or both services) is not a step. The poolmgr update is the first after that start, the
    gateway update the last before it, and a gateway-healthy record must lie between."""
    start = 0
    if release:
        sha = release[:7]
        named = [i for i, x in enumerate(lines) if sha in x]
        check(bool(named), f"the deploy log never names release {sha}")
        applies = [i for i in named if re.search(r"\bapply\b", lines[i], re.I)]
        start = applies[-1] if applies else named[0]

    def step(i: int, svc: str) -> bool:
        x = lines[i]
        if re.search(r"\bplan\b", x, re.I) or ("gateway" in x and "poolmgr" in x):
            return False
        return bool(re.search(rf"\b{svc}\b", x))

    def updates(i: int, svc: str) -> bool:
        return step(i, svc) and bool(_UPDATE.search(lines[i])) and not _HEALTHY.search(lines[i])

    pm = next((i for i in range(start, len(lines)) if updates(i, "poolmgr")), None)
    check(pm is not None, "the deploy log shows no poolmgr update for the release")
    assert pm is not None
    gw = next((i for i in range(pm - 1, start - 1, -1) if updates(i, "gateway")), None)
    check(gw is not None, f"no gateway update before the poolmgr update (line {pm + 1})")
    assert gw is not None
    gh = next(
        (
            i
            for i in range(gw, pm)
            if step(i, "gateway")
            and _HEALTHY.search(lines[i])
            and not _NOT_HEALTHY.search(lines[i])
        ),
        None,
    )
    check(
        gh is not None,
        f"poolmgr updated (line {pm + 1}) with no record that the gateway updated at line "
        f"{gw + 1} was healthy",
    )
    assert gh is not None
    return gw, pm, gh


def run(acc: M3Acceptance, items: list[str], report: Report) -> None:
    order = [i for i in (*BOX_ITEMS, *EXTERNAL_ITEMS) if i in items]
    try:
        for item in order:
            res = ItemResult(item, TITLES[item])
            say(f"\n== {item}: {res.title}")
            t0 = time.monotonic()
            try:
                getattr(acc, "item_" + item.replace("-", "_"))(res)
            except Exception as e:  # recorded, and the run goes on
                res.status = "FAIL"
                res.notes.append(f"{type(e).__name__}: {e}")
            res.seconds = round(time.monotonic() - t0, 1)
            report.items.append(res)
            say(f"   {res.status} ({res.seconds} s) {json.dumps(res.numbers, default=str)}")
            if item == "m3-0" and res.status == "FAIL" and len(order) > 1:
                raise AcceptanceError("the flow failed; the other items need its tenants")
    finally:
        acc.close()


class _External(M3Acceptance):
    """--external: no database, Redis or containers (the operator's machine)."""

    def __init__(self, stack: Stack, report: Report, args: argparse.Namespace) -> None:
        self.stack, self.report, self.args = stack, report, args

    def close(self) -> None:
        pass


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loom-m3-acceptance", description=__doc__.splitlines()[0])
    env = os.environ.get
    p.add_argument("--domain", default=env("LOOM_DOMAIN"), help="the site's domain (§6.11)")
    p.add_argument("--brand-name", default=env("LOOM_BRAND_NAME"), help="the product name (m3-h)")
    p.add_argument("--external", action="store_true", help="only what the internet sees (m3-h)")
    p.add_argument("--only", help=f"comma-separated items, e.g. m3-0,m3-b (of {', '.join(ITEMS)})")
    p.add_argument("--json", type=Path, help="write the report here")
    p.add_argument("--release", default=env("LOOM_RELEASE"), help="the release sha, for the report")
    p.add_argument("--project", default=env("COMPOSE_PROJECT_NAME") or "loom")
    p.add_argument("--database-url", dest="database_url")
    p.add_argument("--redis-url", dest="redis_url")
    p.add_argument("--admin-email", dest="admin_email")
    p.add_argument("--gateway-internal", default="http://gateway:8080")
    p.add_argument("--gateway-metrics", default="http://gateway:9100/metrics")
    p.add_argument("--billing-metrics", default="http://billing:9101/metrics")
    p.add_argument("--webhook-secret-file", help="the endpoint's signing secret (m3-g)")
    p.add_argument(
        "--deploy-log",
        default="/var/lib/loom/deploy.log",
        help="loom-deploy's log, read by m3-p (the default path is an assumption: the spec "
        "does not fix where loom-deploy logs)",
    )
    p.add_argument("--deploy-cmd", default="loom-deploy")
    p.add_argument("--p4-release", help="a release with a newer poolmgr than the gateway's")
    p.add_argument("--log-group", default="/loom/platform")
    p.add_argument("--manual-timeout-s", type=float, default=900)
    p.add_argument("--alarm-wait-s", type=float, default=660)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if not args.domain:
        parser().error("--domain (or LOOM_DOMAIN) is required")
    pool = EXTERNAL_ITEMS if args.external else BOX_ITEMS
    items = [i.strip() for i in args.only.split(",")] if args.only else list(pool)
    unknown = [i for i in items if i not in pool]
    if unknown:
        parser().error(f"not items of this mode: {unknown} (mode items: {', '.join(pool)})")
    stack = public_stack(args)
    report = Report()
    say(f"M3 acceptance ({'external' if args.external else 'box'}) against {stack.web}")
    acc = _External(stack, report, args) if args.external else M3Acceptance(stack, report, args)
    with contextlib.suppress(AcceptanceError):
        run(acc, items, report)
    say("\n== summary")
    for i in report.items:
        say(f"{i.item:5} {i.status:4}  {i.title}")
    if args.json:
        doc = report.to_json()
        doc.update(
            mode="external" if args.external else "box",
            domain=args.domain,
            release=args.release,
            manual={k: v for k, v in MANUAL.items() if k in items},
        )
        args.json.write_text(json.dumps(doc, indent=2, default=str))
    sys.exit(0 if report.passed and report.items else 1)


if __name__ == "__main__":
    main()
