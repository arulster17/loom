"""M1 acceptance on the running compose stack (docs/phase1-m1-acceptance.md).

Drives the real stack from the outside, as a user would: the web app over HTTP exactly
as a browser without JavaScript does (React's progressive-enhancement form posts, so the
same server actions run), the magic link read from Mailpit's API, the gateway with the
unmodified OpenAI Python SDK, the Stripe stub's pay page. Checks every number against
Postgres, Redis and the mock engines' own counters, and records it.

    docker compose up -d --build                     # with LOOM_ADMIN_EMAILS set, below
    uv run loom-m1-acceptance --json /tmp/m1.json    # ~4 minutes; exit 0 = every item passed

The admin is bootstrapped through `LOOM_ADMIN_EMAILS`, which must contain `--admin-email`
(default `admin@m1.test`). Ports default to the compose defaults; a second stack on moved
ports (`-p loom-m1accept`, `LOOM_*_PORT`) is reached with the flags or `LOOM_ACCEPT_*`.
It creates its own users and orgs (random emails), so it can run repeatedly against one
stack; `--only a,b` runs a subset (the flow `0` always runs first, it creates the tenants).

Items (the M1 row of docs/phase1-runpod-plan.md §13 and the brief §8.4):

0  sign up -> pending -> admin approves -> $5 -> key -> SDK (both models, stream and not,
   chat and completions) and the playground -> usage rows and balance drop match
a  usage in Postgres = engine-reported tokens, incl. a client abort and a killed pod
b  a revoked key (and a suspended org) is refused within 5 s
c  concurrent load against a small balance cannot overshoot it
d  gateway p50 added overhead < 10 ms (gateway vs direct to the pod, same mTLS path)
e  loom-loadtest: TTFT and tokens/s through the gateway
f  cold model: 503 model_unavailable + Retry-After, then ready
g  Stripe stub checkout credits exactly once (and a replayed event does not)
h  cross-tenant spot checks, web and API
i  no prompt or completion text (or API key) in any container's logs
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import html
import io
import json
import os
import re
import secrets
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx
import psycopg
import redis

from loom_platform.limiter import org_key

ITEMS = "0abcdefghi"
MODELS = ("qwen3-8b", "llama-3.3-70b-instruct-fp8")
POD_OF = {"qwen3-8b": "mock-pod-1", "llama-3.3-70b-instruct-fp8": "mock-pod-2"}
RAISED_LIMITS = {"rpm_limit": 100_000, "tpm_limit": 1_000_000_000, "concurrency_limit": 256}


class AcceptanceError(AssertionError):
    pass


def check(cond: object, message: str) -> None:
    if not cond:
        raise AcceptanceError(message)


# ---------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class Stack:
    web: str
    gateway: str  # base URL with /v1
    mailpit: str
    database_url: str
    redis_url: str
    project: str
    admin_email: str
    gateway_metrics: str

    @property
    def gateway_root(self) -> str:
        return self.gateway.removesuffix("/v1")


def stack_from_env(args: argparse.Namespace | None = None) -> Stack:
    env = os.environ.get

    def port(var: str, default: int) -> int:
        return int(env(var) or default)

    pg, rds = port("LOOM_PG_PORT", 5432), port("LOOM_REDIS_PORT", 6379)
    defaults = {
        "web": env("LOOM_ACCEPT_WEB_URL") or f"http://localhost:{port('LOOM_WEB_PORT', 3000)}",
        "gateway": env("LOOM_ACCEPT_GATEWAY_URL")
        or f"http://localhost:{port('LOOM_GATEWAY_PORT', 8080)}/v1",
        "mailpit": env("LOOM_ACCEPT_MAILPIT_URL")
        or f"http://localhost:{port('LOOM_MAILPIT_UI_PORT', 8025)}",
        "database_url": env("LOOM_ACCEPT_DATABASE_URL")
        or f"postgresql://loom:loom@localhost:{pg}/loom_platform",
        "redis_url": env("LOOM_ACCEPT_REDIS_URL") or f"redis://localhost:{rds}/0",
        "project": env("COMPOSE_PROJECT_NAME") or "loom",
        "admin_email": "admin@m1.test",
        "gateway_metrics": env("LOOM_ACCEPT_GATEWAY_METRICS_URL")
        or f"http://localhost:{port('LOOM_GATEWAY_METRICS_PORT', 9100)}/metrics",
    }
    if args is not None:
        for k in defaults:
            v = getattr(args, k, None)
            if v:
                defaults[k] = v
    return Stack(**defaults)


# ---------------------------------------------------------------------------- results


@dataclass
class ItemResult:
    item: str
    title: str
    status: str = "PASS"
    numbers: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0


class Report:
    def __init__(self) -> None:
        self.items: list[ItemResult] = []
        self.canaries: set[str] = set()  # prompt markers, completions, keys: never in logs
        self.started = time.time()

    def to_json(self) -> dict[str, Any]:
        return {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "items": [vars(i) for i in self.items],
        }

    @property
    def passed(self) -> bool:
        return all(i.status == "PASS" for i in self.items)


def say(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------- web (no JS)


@dataclass
class Form:
    attrs: dict[str, str]
    fields: list[tuple[str, str, str]]  # name, value, type
    buttons: list[tuple[str, str]]  # name, value

    def value(self, name: str) -> str | None:
        for n, v, _ in self.fields:
            if n == name:
                return v
        return None

    def has(self, *names: str) -> bool:
        return all(self.value(n) is not None for n in names)


class _FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[Form] = []
        self._cur: Form | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if tag == "form":
            self._cur = Form(a, [], [])
            self.forms.append(self._cur)
        elif self._cur is None:
            return
        elif tag == "input" and a.get("name"):
            kind = a.get("type", "text")
            if kind == "checkbox" and "checked" not in a:
                return
            self._cur.fields.append((a["name"], a.get("value", ""), kind))
        elif tag == "button" and a.get("name"):
            self._cur.buttons.append((a["name"], a.get("value", "")))

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._cur = None


def parse_forms(page: str) -> list[Form]:
    p = _FormParser()
    p.feed(page)
    return p.forms


def visible(page: str) -> str:
    """Page text without React's `<!-- -->` separators between text segments."""
    return html.unescape(page.replace("<!-- -->", ""))


def page_errors(page: str) -> list[str]:
    return [
        html.unescape(re.sub(r"<[^>]+>", "", m))
        for m in re.findall(r'class="[^"]*text-red-700[^"]*">(.*?)</p>', page)
    ]


class Browser:
    """A cookie-keeping HTTP client that uses the web app as a no-JS browser would:
    GET pages, submit their server-action forms as multipart posts."""

    def __init__(self, web: str, email: str) -> None:
        self.web = web.rstrip("/")
        self.email = email
        self.http = httpx.Client(
            base_url=self.web, follow_redirects=False, timeout=60, trust_env=False
        )

    def close(self) -> None:
        self.http.close()

    def get(self, path: str, *, follow: bool = True) -> httpx.Response:
        r = self.http.get(path)
        hops = 0
        while follow and r.is_redirect and hops < 8:
            loc = r.headers["location"]
            if loc.startswith("http") and not loc.startswith(self.web):
                return r  # leaving the app (Stripe stub): the caller follows
            r = self.http.get(loc)
            hops += 1
        return r

    def submit(
        self,
        path: str,
        match: Callable[[Form], bool],
        values: dict[str, str] | None = None,
        button: tuple[str, str] | None = None,
        origin: str | None = None,
    ) -> httpx.Response:
        page = self.get(path)
        check(page.status_code == 200, f"GET {path}: {page.status_code}")
        forms = [f for f in parse_forms(page.text) if match(f)]
        check(len(forms) == 1, f"{path}: expected one matching form, found {len(forms)}")
        form = forms[0]
        data: list[tuple[str, str]] = []
        values = dict(values or {})
        for name, value, _ in form.fields:
            data.append((name, values.pop(name, value)))
        data += list(values.items())
        if button is not None:
            check(button in form.buttons, f"{path}: no button {button}")
            data.append(button)
        files = [(k, (None, v)) for k, v in data]
        return self.http.post(
            path,
            files=files,  # type: ignore[arg-type]
            headers={"Origin": origin or self.web, "Accept": "text/html"},
        )


def mailpit_link(stack: Stack, email: str, timeout_s: float = 30) -> str:
    deadline = time.monotonic() + timeout_s
    with httpx.Client(base_url=stack.mailpit, timeout=10, trust_env=False) as mp:
        while time.monotonic() < deadline:
            found = mp.get("/api/v1/search", params={"query": f'to:"{email}"'}).json()
            if found.get("messages"):
                msg = mp.get(f"/api/v1/message/{found['messages'][0]['ID']}").json()
                m = re.search(r"https?://\S+/api/auth/callback/nodemailer\?\S+", msg["Text"])
                check(m, f"no magic link in the email to {email}")
                assert m is not None
                return m.group(0)
            time.sleep(0.25)
    raise AcceptanceError(f"no email to {email} in Mailpit within {timeout_s} s")


def sign_in(stack: Stack, email: str) -> Browser:
    b = Browser(stack.web, email)
    r = b.submit("/signin", lambda f: f.has("email"), {"email": email})
    check(r.status_code in (303, 302), f"sign-in form: {r.status_code} {page_errors(r.text)}")
    check("verify-request" in r.headers.get("location", ""), "sign-in did not ask to check email")
    link = mailpit_link(stack, email)
    check(link.startswith(stack.web), f"magic link {link!r} is not on {stack.web}")
    r = b.http.get(link)
    check(r.is_redirect, f"magic link: {r.status_code}")
    check("authjs.session-token" in b.http.cookies, "no session cookie after the magic link")
    return b


def create_key(b: Browser, name: str) -> str:
    r = b.submit("/dashboard/keys", lambda f: f.has("name", "spend_limit"), {"name": name})
    check(r.status_code == 200, f"create key: {r.status_code}")
    m = re.search(r'data-testid="new-key"[^>]*>(loom_sk_[A-Za-z0-9_]+)<', r.text)
    check(m, f"create key: no key shown; errors {page_errors(r.text)}")
    assert m is not None
    return m.group(1)


# ---------------------------------------------------------------------------- database


class Db:
    def __init__(self, url: str) -> None:
        self.conn = psycopg.connect(url, autocommit=True)

    def one(self, sql: str, *params: Any) -> Any:
        row = self.conn.execute(sql, params).fetchone()  # type: ignore[arg-type]
        return None if row is None else row[0]

    def rows(self, sql: str, *params: Any) -> list[tuple[Any, ...]]:
        return list(self.conn.execute(sql, params).fetchall())  # type: ignore[arg-type]

    def org_of(self, email: str) -> str:
        org = self.one(
            "SELECT m.org_id::text FROM org_members m JOIN users u ON u.id = m.user_id"
            " WHERE u.email = %s",
            email,
        )
        check(org, f"no org for {email}")
        return str(org)

    def key_id(self, plaintext: str) -> str:
        import hashlib

        kid = self.one(
            "SELECT id::text FROM api_keys WHERE key_hash = %s",
            hashlib.sha256(plaintext.encode()).digest(),
        )
        check(kid, "API key not in api_keys")
        return str(kid)

    def balance(self, org: str) -> tuple[int, int, int]:
        """(credit_total, usage_total, posted ledger balance)"""
        row = self.rows(
            "SELECT loom_org_credit_total(%s), loom_org_usage_total(%s), loom_org_balance(%s)",
            org,
            org,
            org,
        )[0]
        return int(row[0]), int(row[1]), int(row[2])

    def usage(self, ids: list[str], timeout_s: float = 30) -> dict[str, dict[str, Any]]:
        """usage_events rows for these request ids, waiting for billing to consume them."""
        deadline = time.monotonic() + timeout_s
        while True:
            rows = self.rows(
                "SELECT id::text, org_id::text, api_key_id::text, model_id, prompt_tokens,"
                " completion_tokens, cached_prompt_tokens, cost_micros, status,"
                " ledger_transaction_id::text, price_version"
                " FROM usage_events WHERE id = ANY(%s::uuid[])",
                ids,
            )
            if len(rows) == len(set(ids)) or time.monotonic() > deadline:
                break
            time.sleep(0.25)
        names = (
            "id org_id api_key_id model_id prompt_tokens completion_tokens cached_prompt_tokens"
            " cost_micros status ledger_transaction_id price_version"
        ).split()
        out = {r[0]: dict(zip(names, r, strict=True)) for r in rows}
        missing = set(ids) - set(out)
        check(not missing, f"usage events never reached Postgres: {sorted(missing)}")
        return out

    def raise_limits(self, org: str) -> None:
        self.conn.execute(
            "UPDATE orgs SET rpm_limit = %(rpm_limit)s, tpm_limit = %(tpm_limit)s,"
            " concurrency_limit = %(concurrency_limit)s WHERE id = %(org)s",
            {**RAISED_LIMITS, "org": org},
        )
        self.conn.execute(
            "UPDATE api_keys SET rpm_limit = %(rpm_limit)s, tpm_limit = %(tpm_limit)s,"
            " concurrency_limit = %(concurrency_limit)s WHERE org_id = %(org)s",
            {**RAISED_LIMITS, "org": org},
        )


# ---------------------------------------------------------------------------- docker


def container(stack: Stack, service: str) -> str:
    out = subprocess.run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label=com.docker.compose.project={stack.project}",
            "--filter",
            f"label=com.docker.compose.service={service}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    check(len(out) == 1, f"compose service {service} of project {stack.project}: {out}")
    return out[0]


def docker(*args: str, input_text: str | None = None, timeout: float = 600) -> str:
    r = subprocess.run(
        ["docker", *args], input=input_text, capture_output=True, text=True, timeout=timeout
    )
    if r.returncode != 0:
        raise AcceptanceError(f"docker {' '.join(args[:3])}: {r.returncode} {r.stderr[-500:]}")
    return r.stdout


def wait_healthy(cid: str, timeout_s: float = 90) -> float:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        state = docker("inspect", "-f", "{{.State.Health.Status}}", cid).strip()
        if state == "healthy":
            return time.monotonic() - t0
        time.sleep(0.5)
    raise AcceptanceError(f"container {cid} not healthy within {timeout_s} s")


def engine_counters(stack: Stack, model: str) -> tuple[int, int]:
    """(prompt_tokens_total, generation_tokens_total) of the mock engine of `model`, read
    on the pod itself from the engine's own Prometheus endpoint."""
    code = (
        "import urllib.request;"
        "print(urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5).read().decode())"
    )
    text = docker("exec", container(stack, POD_OF[model]), "python", "-c", code)
    vals = {}
    for name in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total"):
        total = 0.0
        for line in text.splitlines():
            if line.startswith(name):
                total += float(line.rsplit(" ", 1)[1])
        vals[name] = int(total)
    return vals["vllm:prompt_tokens_total"], vals["vllm:generation_tokens_total"]


def poolsim(stack: Stack, *args: str) -> str:
    return docker("exec", container(stack, "poolsim"), "loom-poolsim", *args)


# ---------------------------------------------------------------------------- API helpers


def openai_client(stack: Stack, key: str) -> Any:
    from openai import OpenAI

    return OpenAI(base_url=stack.gateway, api_key=key, max_retries=0, timeout=60)


def canary() -> str:
    return f"m1canary{secrets.token_hex(8)}"


@dataclass
class Call:
    request_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_micros: int
    text: str
    status: str = "ok"


def sdk_chat(client: Any, model: str, prompt: str, *, stream: bool, max_tokens: int = 24) -> Call:
    msgs = [{"role": "user", "content": prompt}]
    if not stream:
        raw = client.chat.completions.with_raw_response.create(
            model=model, messages=msgs, max_tokens=max_tokens
        )
        resp = raw.parse()
        u = resp.usage
        cost = int(raw.headers["x-loom-cost-micros"])
        check(cost == u.model_extra.get("cost_micros"), "cost header != usage.cost_micros")
        cached = (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
        return Call(
            raw.headers["x-loom-request-id"],
            model,
            u.prompt_tokens,
            u.completion_tokens,
            cached,
            cost,
            resp.choices[0].message.content or "",
        )
    raw = client.chat.completions.with_raw_response.create(
        model=model,
        messages=msgs,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    text, usage = [], None
    for chunk in raw.parse():
        if chunk.choices and chunk.choices[0].delta.content:
            text.append(chunk.choices[0].delta.content)
        if chunk.usage is not None:
            usage = chunk.usage
    check(usage is not None, "stream ended without a usage chunk")
    assert usage is not None
    cached = (usage.prompt_tokens_details.cached_tokens or 0) if usage.prompt_tokens_details else 0
    return Call(
        raw.headers["x-loom-request-id"],
        model,
        usage.prompt_tokens,
        usage.completion_tokens,
        cached,
        int(usage.model_extra["cost_micros"]),
        "".join(text),
    )


def sdk_completion(client: Any, model: str, prompt: str) -> Call:
    raw = client.completions.with_raw_response.create(model=model, prompt=prompt)
    resp = raw.parse()
    u = resp.usage
    return Call(
        raw.headers["x-loom-request-id"],
        model,
        u.prompt_tokens,
        u.completion_tokens,
        0,
        int(raw.headers["x-loom-cost-micros"]),
        resp.choices[0].text,
    )


def sse_events(lines: Iterator[str]) -> Iterator[dict[str, Any] | str]:
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            yield "[DONE]"
            return
        yield json.loads(data)


def check_rows(calls: list[Call], rows: dict[str, dict[str, Any]], org: str) -> None:
    for c in calls:
        r = rows[c.request_id]
        got = (r["prompt_tokens"], r["completion_tokens"], r["cached_prompt_tokens"])
        want = (c.prompt_tokens, c.completion_tokens, c.cached_tokens)
        check(got == want, f"{c.request_id}: usage_events {got} != engine usage {want}")
        check(r["cost_micros"] == c.cost_micros, f"{c.request_id}: cost {r['cost_micros']}")
        check(r["org_id"] == org, f"{c.request_id}: billed to {r['org_id']}, not {org}")
        check(r["status"] == c.status, f"{c.request_id}: status {r['status']} != {c.status}")


def wait_posted(db: Db, ids: list[str], timeout_s: float = 140) -> float:
    """Wait for the per-minute batch to post these events to the ledger."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        n = db.one(
            "SELECT count(*) FROM usage_events WHERE id = ANY(%s::uuid[])"
            " AND (ledger_transaction_id IS NOT NULL OR cost_micros = 0)",
            ids,
        )
        if n == len(set(ids)):
            return time.monotonic() - t0
        time.sleep(1)
    raise AcceptanceError(f"usage not posted to the ledger within {timeout_s} s")


def usd_exact(micros: int) -> str:
    """The web app's formatUsdExact."""
    whole, frac = divmod(abs(micros), 1_000_000)
    f = f"{frac:06d}".rstrip("0")
    f = f.ljust(2, "0")
    return f"{'-' if micros < 0 else ''}${whole:,}.{f}"


def usd_cents(micros: int) -> str:
    units = (abs(micros) + 5_000) // 10_000
    return f"{'-' if micros < 0 and units else ''}${units // 100:,}.{units % 100:02d}"


# ---------------------------------------------------------------------------- the run


class Acceptance:
    def __init__(self, stack: Stack, report: Report) -> None:
        self.stack = stack
        self.report = report
        self.db = Db(stack.database_url)
        self.redis = redis.Redis.from_url(stack.redis_url)
        self.run_id = secrets.token_hex(3)
        self.browsers: dict[str, Browser] = {}
        self.admin: Browser | None = None

    def email(self, who: str) -> str:
        return f"{who}-{self.run_id}@m1.test"

    def close(self) -> None:
        for b in self.browsers.values():
            b.close()
        if self.admin:
            self.admin.close()
        self.db.conn.close()
        self.redis.close()

    def prompt(self, extra: str = "") -> str:
        c = canary()
        self.report.canaries.add(c)
        return f"Say something about {c}. {extra}".strip()

    def remember(self, call: Call) -> Call:
        # Completions as log canaries: long enough not to occur by chance.
        if len(call.text) >= 24:
            self.report.canaries.add(call.text[:48])
        return call

    # -- tenants

    def signup_and_approve(self, who: str) -> tuple[Browser, str]:
        email = self.email(who)
        b = sign_in(self.stack, email)
        self.browsers[who] = b
        org = self.db.org_of(email)
        r = b.get("/dashboard")
        check(
            r.status_code == 200 and "awaiting approval" in r.text,
            f"{who}: a new signup should see the pending page",
        )
        check(
            self.db.one("SELECT status FROM orgs WHERE id = %s", org) == "pending_approval",
            "new org is not pending_approval",
        )
        # A pending org cannot create keys.
        rk = b.get("/dashboard/keys")
        check("awaiting approval" in rk.text, "a pending org reached the keys page")
        self.approve(org)
        return b, org

    def admin_browser(self) -> Browser:
        if self.admin is None:
            self.admin = sign_in(self.stack, self.stack.admin_email)
            r = self.admin.get("/admin")
            check(
                r.status_code == 200 and "Approve signups" in r.text,
                f"{self.stack.admin_email} is not an admin: is it in LOOM_ADMIN_EMAILS?",
            )
        return self.admin

    def approve(self, org: str) -> None:
        a = self.admin_browser()
        r = a.submit(
            "/admin",
            lambda f: f.value("org_id") == org and ("op", "approve") in f.buttons,
            button=("op", "approve"),
        )
        check(r.status_code == 200, f"approve: {r.status_code}")
        check("credit added" in r.text, f"approve: {page_errors(r.text)}")

    # -- item 0: the flow

    def flow(self, res: ItemResult) -> None:
        b, org = self.signup_and_approve("alice")
        self.alice, self.alice_org = b, org
        credit, usage, posted = self.db.balance(org)
        check((credit, usage, posted) == (5_000_000, 0, 5_000_000), f"after approval {credit}")
        page = visible(b.get("/dashboard").text)
        check(">$5.00<" in page, "dashboard does not show $5.00")
        promo = self.db.rows(
            "SELECT kind, idempotency_key FROM ledger_transactions WHERE org_id = %s", org
        )
        check(promo == [("promo", f"promo:signup:{org}")], f"ledger after approval: {promo}")
        res.numbers["signup_credit_micros"] = credit

        key = create_key(b, "acceptance")
        self.alice_key = key
        self.report.canaries.add(key)
        check("acceptance" in b.get("/dashboard/keys").text, "key not listed")

        client = openai_client(self.stack, key)
        models = [m.id for m in client.models.list().data]
        check(set(MODELS) <= set(models), f"/v1/models: {models}")
        before = {m: engine_counters(self.stack, m) for m in MODELS}
        calls: list[Call] = []
        for model in MODELS:
            calls.append(self.remember(sdk_chat(client, model, self.prompt(), stream=False)))
            calls.append(self.remember(sdk_chat(client, model, self.prompt(), stream=True)))
        calls.append(self.remember(sdk_completion(client, MODELS[0], self.prompt())))
        check(calls[-1].completion_tokens <= 16, "/v1/completions: more than the default 16 tokens")
        after = {m: engine_counters(self.stack, m) for m in MODELS}

        # The playground: the browser's session, the server's playground key.
        pg_call = self.playground(b, MODELS[0])
        calls.append(pg_call)
        rows = self.db.usage([c.request_id for c in calls])
        check_rows(calls, rows, org)
        pg_key = rows[pg_call.request_id]["api_key_id"]
        scopes = self.db.one("SELECT scopes FROM api_keys WHERE id = %s", pg_key)
        check(scopes == ["playground"], f"playground request used key with scopes {scopes}")

        # Engine counters: every token the engines report is in usage_events, and no more.
        for m in MODELS:
            dp = after[m][0] - before[m][0]
            dg = after[m][1] - before[m][1]
            mine = [c for c in calls[:-1] if c.model == m]
            sp = sum(c.prompt_tokens for c in mine)
            sg = sum(c.completion_tokens for c in mine)
            check((dp, dg) == (sp, sg), f"{m}: engine counted {(dp, dg)}, billed {(sp, sg)}")
            res.numbers[f"{m}_engine_vs_db_tokens"] = f"prompt {dp}={sp}, completion {dg}={sg}"

        total = sum(c.cost_micros for c in calls)
        credit, usage, _ = self.db.balance(org)
        check(usage == total, f"usage total {usage} != sum of request costs {total}")
        bill = visible(b.get("/dashboard/billing").text)
        check(
            usd_exact(total) + " used" in bill,
            f"billing page does not show {usd_exact(total)} used",
        )
        check(usd_cents(credit - usage) in bill, "billing page balance")
        # Redis agrees with Postgres.
        check(
            int(self.redis.get(org_key(org, "usage")) or -1) == usage,
            "Redis usage != Postgres usage",
        )
        # The usage CSV adds up to the same.
        csv_rows = list(csv.DictReader(io.StringIO(b.get("/api/usage.csv").text)))
        check(sum(int(r["requests"]) for r in csv_rows) == len(calls), "CSV request count")
        check(
            sum(int(r["completion_tokens"]) for r in csv_rows)
            == sum(c.completion_tokens for c in calls),
            "CSV completion tokens",
        )
        check({r["key_name"] for r in csv_rows} == {"acceptance", "Playground"}, "CSV key names")
        # The per-minute batch posts it to the ledger; the posted balance drops exactly.
        waited = wait_posted(self.db, [c.request_id for c in calls])
        _, _, posted = self.db.balance(org)
        check(posted == 5_000_000 - total, f"ledger balance {posted} != {5_000_000 - total}")
        res.numbers.update(
            requests=len(calls),
            cost_micros=total,
            balance_after_micros=posted,
            ledger_posted_after_s=round(waited, 1),
            web_shows=f"{usd_exact(total)} used, balance {usd_cents(credit - usage)}",
        )
        self.flow_calls = calls

    def playground(self, b: Browser, model: str) -> Call:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": self.prompt()}],
            "temperature": 0.7,
            "top_p": 1,
            "max_tokens": 24,
        }
        evil = b.http.post(
            "/api/playground/chat", json=body, headers={"Origin": "http://evil.example"}
        )
        check(evil.status_code == 403, f"cross-site playground post: {evil.status_code}")
        usage, text = None, []
        with b.http.stream(
            "POST", "/api/playground/chat", json=body, headers={"Origin": b.web}
        ) as r:
            check(r.status_code == 200, f"playground: {r.status_code}")
            rid = r.headers["x-loom-request-id"]
            for ev in sse_events(r.iter_lines()):
                if isinstance(ev, dict):
                    if ev.get("usage"):
                        usage = ev["usage"]
                    for ch in ev.get("choices") or []:
                        text.append(ch.get("delta", {}).get("content") or "")
        check(usage is not None, "playground stream had no usage chunk")
        assert usage is not None
        return self.remember(
            Call(
                rid,
                model,
                usage["prompt_tokens"],
                usage["completion_tokens"],
                0,
                usage["cost_micros"],
                "".join(text),
            )
        )

    # -- item a: aborted streams

    def item_a(self, res: ItemResult) -> None:
        key, org = self.alice_key, self.alice_org
        body = {
            "model": MODELS[0],
            "messages": [{"role": "user", "content": self.prompt()}],
            "max_tokens": 400,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
        hdr = {"Authorization": f"Bearer {key}"}
        # 1. Client abort: read 10 content chunks, then hang up.
        p0, g0 = engine_counters(self.stack, MODELS[0])
        seen = 0
        last_seen: dict[str, Any] | None = None
        with (
            httpx.Client(timeout=60, trust_env=False) as http,
            http.stream(
                "POST", f"{self.stack.gateway}/chat/completions", json=body, headers=hdr
            ) as r,
        ):
            check(r.status_code == 200, f"abort stream: {r.status_code}")
            rid = r.headers["x-loom-request-id"]
            for ev in sse_events(r.iter_lines()):
                if isinstance(ev, dict) and ev.get("choices"):
                    last_seen = ev.get("usage")
                    if ev["choices"][0].get("delta", {}).get("content"):
                        seen += 1
                if seen >= 10:
                    break
        time.sleep(1.0)  # let the engine see the hang-up
        p1, g1 = engine_counters(self.stack, MODELS[0])
        row = self.db.usage([rid])[rid]
        check(row["status"] == "aborted", f"client abort: status {row['status']}")
        check(last_seen is not None, "no per-chunk usage")
        assert last_seen is not None
        check(
            row["prompt_tokens"] == p1 - p0,
            f"client abort: prompt {row['prompt_tokens']} != engine {p1 - p0}",
        )
        check(
            last_seen["completion_tokens"] <= row["completion_tokens"] <= g1 - g0,
            f"client abort: billed {row['completion_tokens']} outside "
            f"[client saw {last_seen['completion_tokens']}, engine generated {g1 - g0}]",
        )
        res.numbers["client_abort"] = {
            "client_received_completion_tokens": last_seen["completion_tokens"],
            "billed_completion_tokens": row["completion_tokens"],
            "engine_generated_completion_tokens": g1 - g0,
            "billed_prompt_tokens": row["prompt_tokens"],
            "engine_prompt_tokens": p1 - p0,
            "cost_micros": row["cost_micros"],
        }
        if row["completion_tokens"] != g1 - g0:
            res.notes.append(
                "client abort: the engine generated "
                f"{g1 - g0 - row['completion_tokens']} token(s) after the last chunk the "
                "gateway forwarded; they are not billed (contracts §9: bill what was sent)"
            )

        # 2. Pod killed mid-stream: the client reads to the end and sees upstream_lost.
        model = MODELS[1]
        pod = container(self.stack, POD_OF[model])
        body2 = {**body, "model": model, "messages": [{"role": "user", "content": self.prompt()}]}
        events: list[dict[str, Any] | str] = []
        killed_at: float | None = None
        with (
            httpx.Client(timeout=60, trust_env=False) as http,
            http.stream(
                "POST", f"{self.stack.gateway}/chat/completions", json=body2, headers=hdr
            ) as r,
        ):
            check(r.status_code == 200, f"kill stream: {r.status_code}")
            rid2 = r.headers["x-loom-request-id"]
            content = 0
            for ev in sse_events(r.iter_lines()):
                events.append(ev)
                if isinstance(ev, dict) and ev.get("choices"):
                    if ev["choices"][0].get("delta", {}).get("content"):
                        content += 1
                    if content == 10 and killed_at is None:
                        killed_at = time.monotonic()
                        docker("kill", pod)
        check(events and events[-1] == "[DONE]", "kill: stream did not end with [DONE]")
        errs = [e for e in events if isinstance(e, dict) and "error" in e]
        check(
            errs and errs[-1]["error"]["code"] == "upstream_lost",
            f"kill: no upstream_lost error chunk: {errs}",
        )
        usages = [e["usage"] for e in events if isinstance(e, dict) and e.get("usage")]
        last = usages[-1]
        row2 = self.db.usage([rid2])[rid2]
        check(row2["status"] == "aborted", f"pod kill: status {row2['status']}")
        got = (row2["prompt_tokens"], row2["completion_tokens"])
        want = (last["prompt_tokens"], last["completion_tokens"])
        check(got == want, f"pod kill: billed {got} != last engine usage the client saw {want}")
        docker("start", pod)
        back = wait_healthy(pod)
        # The replica serves again (poolsim keeps it ready; the gateway retries it).
        call = sdk_chat(openai_client(self.stack, key), model, self.prompt(), stream=False)
        self.remember(call)
        res.numbers["pod_kill"] = {
            "engine_completion_tokens_last_chunk": last["completion_tokens"],
            "billed_completion_tokens": row2["completion_tokens"],
            "billed_prompt_tokens": row2["prompt_tokens"],
            "client_saw_error": errs[-1]["error"]["code"],
            "pod_healthy_again_after_s": round(back, 1),
        }
        rows = self.db.usage([call.request_id])
        check_rows([call], rows, org)

    # -- item b: revocation

    def item_b(self, res: ItemResult) -> None:
        b = self.alice
        key = create_key(b, "to-revoke")
        self.report.canaries.add(key)
        kid = self.db.key_id(key)
        url = f"{self.stack.gateway}/chat/completions"
        hdr = {"Authorization": f"Bearer {key}"}
        with httpx.Client(timeout=10, trust_env=False) as http:
            # `{}` passes auth and fails validation (400): free, and fast.
            for _ in range(5):
                r = http.post(url, json={}, headers=hdr)
                check(r.status_code == 400, f"warm-up with a live key: {r.status_code}")
            ok = http.post(
                url, json={"model": MODELS[0], "messages": [], "max_tokens": 1}, headers=hdr
            )
            check(ok.status_code in (200, 400), f"live key: {ok.status_code}")
            t_click = time.monotonic()
            rr = b.submit("/dashboard/keys", lambda f: f.value("key_id") == kid)
            t_done = time.monotonic()
            check(rr.status_code == 200 and not page_errors(rr.text), "revoke failed")
            first_401 = None
            statuses: Counter[int] = Counter()
            while time.monotonic() - t_done < 6:
                s = http.post(url, json={}, headers=hdr).status_code
                statuses[s] += 1
                if s == 401 and first_401 is None:
                    first_401 = time.monotonic()
                if first_401 is not None and s != 401:
                    raise AcceptanceError("revoked key accepted again after a 401")
                if first_401 is not None and time.monotonic() - first_401 > 1:
                    break
                time.sleep(0.01)
        check(first_401 is not None, "revoked key still accepted after 6 s")
        assert first_401 is not None
        since_commit = first_401 - t_done
        since_click = first_401 - t_click
        check(since_click < 5, f"revocation took {since_click:.3f} s")
        res.numbers["revoked_key_refused_after_s"] = {
            "from_revoke_response": round(max(since_commit, 0.0), 3),
            "from_revoke_click": round(since_click, 3),
            "revoke_request_s": round(t_done - t_click, 3),
        }
        res.numbers["polls"] = dict(statuses)

        # Suspension: the whole org's keys stop within 5 s too (then reactivate).
        b2, org2 = self.signup_and_approve("sus")
        key2 = create_key(b2, "k")
        self.report.canaries.add(key2)
        hdr2 = {"Authorization": f"Bearer {key2}"}
        a = self.admin_browser()
        with httpx.Client(timeout=10, trust_env=False) as http:
            check(http.post(url, json={}, headers=hdr2).status_code == 400, "live key")
            t0 = time.monotonic()
            r = a.submit(
                f"/admin/orgs/{org2}",
                lambda f: f.value("org_id") == org2 and ("op", "suspend") in f.buttons,
                {"reason": "M1 acceptance"},
                button=("op", "suspend"),
            )
            check("Suspended" in r.text, f"suspend: {page_errors(r.text)}")
            t1 = None
            while time.monotonic() - t0 < 6:
                resp = http.post(url, json={}, headers=hdr2)
                if resp.status_code == 403:
                    check(resp.json()["error"]["code"] == "org_suspended", "suspended code")
                    t1 = time.monotonic()
                    break
                time.sleep(0.01)
        check(t1 is not None and t1 - t0 < 5, "suspended org's key not refused within 5 s")
        assert t1 is not None
        res.numbers["suspended_org_refused_after_s"] = round(t1 - t0, 3)

    # -- item c: overshoot

    def item_c(self, res: ItemResult) -> None:
        b, org = self.signup_and_approve("broke")
        key = create_key(b, "load")
        self.report.canaries.add(key)
        # Leave $0.002 of the $5: an admin correction through the admin page.
        a = self.admin_browser()
        r = a.submit(
            f"/admin/orgs/{org}",
            lambda f: f.has("amount", "memo"),
            {"amount": "-4.998", "memo": "M1 acceptance: small balance"},
        )
        check("Removed $5.00" in r.text or "Removed" in r.text, f"adjust: {page_errors(r.text)}")
        credit, usage, _ = self.db.balance(org)
        check(credit - usage == 2_000, f"balance {credit - usage}, wanted 2000 micros")
        # Raise the org's and key's limits so money is the only thing that stops requests.
        self.db.raise_limits(org)
        time.sleep(0.5)

        prompt = self.prompt()
        url = f"{self.stack.gateway}/chat/completions"
        hdr = {"Authorization": f"Bearer {key}"}
        statuses: Counter[str] = Counter()
        ids: list[str] = []
        max_costs: list[int] = []
        min_available = [credit - usage]
        stop = asyncio.Event()

        async def worker(http: httpx.AsyncClient, i: int) -> None:
            n = 0
            while not stop.is_set() and n < 40:
                n += 1
                model = MODELS[(i + n) % 2]
                stream = (i + n) % 3 == 0
                body: dict[str, Any] = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 48,
                    "stream": stream,
                }
                if stream:
                    body["stream_options"] = {"include_usage": True}
                async with http.stream("POST", url, json=body, headers=hdr) as resp:
                    statuses[str(resp.status_code)] += 1
                    if resp.status_code == 200:
                        ids.append(resp.headers["x-loom-request-id"])
                        max_costs.append(int(resp.headers["x-loom-max-cost-micros"]))
                        # A third of the streams hang up early (aborted, billed partially).
                        if stream and n % 2 == 0:
                            async for _ in resp.aiter_lines():
                                break
                            continue
                    await resp.aread()

        async def sampler() -> None:
            names = [org_key(org, n) for n in ("credit", "usage", "rsum")]
            while not stop.is_set():
                c, u, rs = self.redis.mget(names)
                if c is not None and u is not None:
                    min_available.append(int(c) - int(u) - int(rs or 0))
                await asyncio.sleep(0.005)

        async def run() -> float:
            limits = httpx.Limits(max_connections=96, max_keepalive_connections=96)
            async with httpx.AsyncClient(timeout=60, limits=limits, trust_env=False) as http:
                t0 = time.monotonic()
                s = asyncio.create_task(sampler())
                await asyncio.gather(*(worker(http, i) for i in range(64)))
                stop.set()
                await s
                return time.monotonic() - t0

        elapsed = asyncio.run(run())
        rows = self.db.usage(ids, timeout_s=60)
        spent = sum(r["cost_micros"] for r in rows.values())
        credit, usage, _ = self.db.balance(org)
        overshoot = max(0, usage - credit)
        check(usage == spent, "usage_total != the sum of the admitted requests' costs")
        check(statuses.get("402", 0) > 0, "the balance never ran out: test too small")
        check(overshoot == 0, f"overshoot {overshoot} micros")
        check(min(min_available) >= 0, f"Redis available went to {min(min_available)}")
        waited = wait_posted(self.db, ids)
        _, _, posted = self.db.balance(org)
        check(posted >= 0, f"posted ledger balance {posted} < 0")
        res.numbers.update(
            start_balance_micros=2_000,
            concurrency=64,
            attempts=sum(statuses.values()),
            http_status=dict(statuses),
            admitted=len(ids),
            aborted=sum(1 for r in rows.values() if r["status"] == "aborted"),
            spent_micros=spent,
            final_balance_micros=credit - usage,
            posted_ledger_balance_micros=posted,
            min_redis_available_micros=min(min_available),
            worst_case_overshoot_micros=overshoot,
            documented_bound_micros=0,
            reservation_micros_range=[min(max_costs), max(max_costs)],
            seconds=round(elapsed, 1),
            ledger_posted_after_s=round(waited, 1),
        )

    # -- item d: overhead

    def item_d(self, res: ItemResult) -> None:
        key = self.load_keys()[0]
        probe = (Path(__file__).parent / "overhead_probe.py").read_text()
        out = docker(
            "exec",
            "-i",
            container(self.stack, "poolsim"),
            "python",
            "-",
            "--api-key",
            key,
            "--pairs",
            "400",
            input_text=probe,
        )
        result = json.loads(out.strip().splitlines()[-1])
        # The gateway's own view: time from receipt to dispatch and the settle call.
        metrics = httpx.get(self.stack.gateway_metrics, timeout=10).text

        def mean(phase: str) -> float | None:
            s = re.search(rf'loom_gateway_overhead_seconds_sum\{{phase="{phase}"\}} (\S+)', metrics)
            n = re.search(
                rf'loom_gateway_overhead_seconds_count\{{phase="{phase}"\}} (\S+)', metrics
            )
            if not s or not n or float(n.group(1)) == 0:
                return None
            return round(float(s.group(1)) / float(n.group(1)) * 1000, 3)

        result = {k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()}
        result["in_gateway_pre_dispatch_mean_ms"] = mean("pre_dispatch")
        result["in_gateway_settle_mean_ms"] = mean("settle")
        res.numbers.update(result)
        check(result["overhead_p50_ms"] < 10, f"p50 overhead {result['overhead_p50_ms']} ms")

    # -- item e: load test

    def load_keys(self) -> list[str]:
        if getattr(self, "_load_keys", None) is None:
            b, org = self.signup_and_approve("load")
            keys = [create_key(b, f"load-{i}") for i in range(4)]
            self.report.canaries.update(keys)
            self.db.raise_limits(org)
            self._load_keys, self.load_org = keys, org
        return self._load_keys

    def item_e(self, res: ItemResult) -> None:
        from loom_devstack.loadtest import LoadTestConfig, render, run_load
        from loom_devstack.loadtest import report as lt_report

        keys = self.load_keys()
        for model, conc, n in ((MODELS[0], 32, 400), (MODELS[1], 16, 200)):
            cfg = LoadTestConfig(
                base_url=self.stack.gateway,
                api_keys=tuple(keys),
                model=model,
                concurrency=conc,
                requests=n,
                warmup=8,
            )
            rep = lt_report(asyncio.run(run_load(cfg)))
            say(render(rep, cfg))
            s = rep.summary
            res.numbers[model] = {
                "concurrency": conc,
                "requests": s.n_total,
                "http_status": rep.http_status,
                "error_rate": s.error_rate,
                "ttft_ms_p50": round(s.ttft_ms.p50 or 0, 1),
                "ttft_ms_p95": round(s.ttft_ms.p95 or 0, 1),
                "tpot_ms_p50": round(s.tpot_ms.p50 or 0, 2),
                "per_request_output_tok_s_p50": round(s.per_request_output_tok_s.p50 or 0, 1),
                "aggregate_output_tok_s": round(s.throughput.output_tok_s, 1),
                "request_rate_per_s": round(s.throughput.request_rate, 2),
            }
            check(s.error_rate == 0, f"{model}: load test error rate {s.error_rate}")
            check(rep.http_status == {"200": n}, f"{model}: statuses {rep.http_status}")

    # -- item f: cold model

    def item_f(self, res: ItemResult) -> None:
        client_key = self.alice_key
        model = MODELS[0]
        hdr = {"Authorization": f"Bearer {client_key}"}
        url = f"{self.stack.gateway}/chat/completions"
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        poolsim(self.stack, "set", model, "cold")
        try:
            t0 = time.monotonic()
            with httpx.Client(timeout=30, trust_env=False) as http:
                r = http.post(url, json=body, headers=hdr)
                check(r.status_code == 503, f"cold: {r.status_code}")
                err = r.json()["error"]
                check(err["code"] == "model_unavailable", f"cold: code {err['code']}")
                check("temporarily unavailable" in err["message"], f"message {err['message']!r}")
                retry_after = int(r.headers["retry-after"])
                rid = r.headers["x-loom-request-id"]
                states = []
                seen = None
                while time.monotonic() - t0 < 90:
                    r2 = http.post(url, json=body, headers=hdr)
                    if r2.status_code == 200:
                        break
                    check(r2.status_code == 503, f"warming: {r2.status_code}")
                    ra = int(r2.headers["retry-after"])
                    if seen != ra:
                        states.append(ra)
                        seen = ra
                    time.sleep(1)
                check(r2.status_code == 200, "model never became ready")
                ready_after = time.monotonic() - t0
            check(
                self.db.one("SELECT count(*) FROM usage_events WHERE id = %s", rid) == 0,
                "a 503 before reserve must not emit a usage event",
            )
        finally:
            poolsim(self.stack, "set", model, "ready")
        res.numbers.update(
            first_status=503,
            message=err["message"],
            retry_after_s=retry_after,
            retry_after_while_warming=states[:8],
            ready_after_s=round(ready_after, 1),
        )

    # -- item g: Stripe stub

    def item_g(self, res: ItemResult) -> None:
        b, org = self.alice, self.alice_org
        credit0, _, _ = self.db.balance(org)
        r = b.submit(
            "/dashboard/billing",
            lambda f: f.has("amount") and not f.has("threshold"),
            {"amount": "10"},
        )
        check(r.status_code in (303, 302), f"add credit: {r.status_code} {page_errors(r.text)}")
        pay = r.headers["location"]
        check("/checkout/cs_" in pay, f"checkout redirect {pay!r}")
        sid = pay.rsplit("/", 1)[1]
        with httpx.Client(timeout=30, trust_env=False, follow_redirects=False) as stub:
            page = stub.get(pay)
            check(page.status_code == 200 and "Pay" in page.text, "stub pay page")
            done = stub.post(f"{pay}/complete")
            check(done.is_redirect, f"pay: {done.status_code}")
            check("checkout=success" in done.headers["location"], "pay did not return to the app")

            def purchases() -> list[tuple[Any, ...]]:
                return self.db.rows(
                    "SELECT id, idempotency_key, external_ref FROM ledger_transactions"
                    " WHERE org_id = %s AND kind = 'purchase'",
                    org,
                )

            deadline = time.monotonic() + 15
            while not purchases() and time.monotonic() < deadline:
                time.sleep(0.2)
            first = purchases()
            check(len(first) == 1, f"purchases after paying: {first}")
            # Pay again: the stub re-sends the same event id (a replay).
            stub.post(f"{pay}/complete")
            time.sleep(2)
        after = purchases()
        credit1, _, _ = self.db.balance(org)
        check(after == first, f"a replayed webhook credited again: {after}")
        check(credit1 - credit0 == 10_000_000, f"credit rose by {credit1 - credit0}")
        events = self.db.one(
            "SELECT count(*) FROM stripe_events WHERE payload::text LIKE %s", f"%{sid}%"
        )
        bill = visible(b.get("/dashboard/billing?checkout=success").text)
        check("$10.00" in bill, "billing page does not list the $10 purchase")
        res.numbers.update(
            amount_micros=credit1 - credit0,
            purchase_transactions=len(after),
            idempotency_key=after[0][1],
            stripe_events_stored=events,
        )

    # -- item h: cross-tenant

    def item_h(self, res: ItemResult) -> None:
        a_org, a_key = self.alice_org, self.alice_key
        bob, b_org = self.signup_and_approve("bob")
        b_key = create_key(bob, "bobs")
        self.report.canaries.add(b_key)
        a_kid = self.db.key_id(a_key)
        checks: dict[str, str] = {}

        # Web: Bob cannot revoke Alice's key by posting her key id with his session.
        page = bob.get("/dashboard/keys")
        forms = [f for f in parse_forms(page.text) if f.has("key_id")]
        check(forms, "Bob has no revoke form to borrow")
        form = forms[0]
        data = [(n, a_kid if n == "key_id" else v) for n, v, _ in form.fields]
        r = bob.http.post(
            "/dashboard/keys",
            files=[(k, (None, v)) for k, v in data],  # type: ignore[arg-type]
            headers={"Origin": bob.web},
        )
        errs = page_errors(r.text)
        check(errs, "revoking another org's key showed no error")
        check(
            self.db.one("SELECT revoked_at FROM api_keys WHERE id = %s", a_kid) is None,
            "Bob revoked Alice's key",
        )
        checks["revoke other org's key"] = f"refused: {errs[0]!r}"
        # Admin pages are a plain 404 for non-admins.
        for path in ("/admin", f"/admin/orgs/{a_org}"):
            s = bob.get(path).status_code
            check(s == 404, f"Bob GET {path}: {s}")
            checks[f"GET {path}"] = str(s)
        # Bob's pages and CSV show nothing of Alice's.
        a_prefix = a_key[:20]
        for path in ("/dashboard/keys", "/dashboard/usage", "/api/usage.csv", "/dashboard"):
            text = bob.get(path).text
            check(a_prefix not in text and "acceptance" not in text, f"{path} leaks Alice's key")
        checks["Bob's dashboard/keys/usage/CSV"] = "no trace of Alice's keys"
        # API: Bob's key bills Bob, not Alice; Alice's totals do not move.
        _, a_usage0, _ = self.db.balance(a_org)
        call = self.remember(
            sdk_chat(openai_client(self.stack, b_key), MODELS[0], self.prompt(), stream=False)
        )
        row = self.db.usage([call.request_id])[call.request_id]
        check(row["org_id"] == b_org, "Bob's request billed to another org")
        _, a_usage1, _ = self.db.balance(a_org)
        check(a_usage0 == a_usage1, "Alice's usage moved on Bob's request")
        checks["Bob's API request"] = "billed to Bob's org; Alice's usage unchanged"
        # A tampered key is refused; a pending org cannot reach its keys page.
        pend = sign_in(self.stack, self.email("pending"))
        self.browsers["pending"] = pend
        p_org = self.db.org_of(self.email("pending"))
        r2 = httpx.post(
            f"{self.stack.gateway}/chat/completions",
            json={},
            headers={"Authorization": f"Bearer {a_key[:-1]}x"},
            timeout=10,
        )
        check(r2.status_code == 401, f"tampered key: {r2.status_code}")
        checks["tampered key"] = "401"
        check(
            "awaiting approval" in pend.get("/dashboard/keys").text,
            "pending user reached the keys page",
        )
        checks["pending org keys page"] = "redirected to the approval page"
        res.numbers.update(checks)
        res.numbers["orgs"] = {"alice": a_org, "bob": b_org, "pending": p_org}

    # -- item i: logs

    def item_i(self, res: ItemResult) -> None:
        ids = docker(
            "ps", "-aq", "--filter", f"label=com.docker.compose.project={self.stack.project}"
        ).split()
        hits: dict[str, list[str]] = {}
        lines = 0
        for cid in ids:
            r = subprocess.run(["docker", "logs", cid], capture_output=True, text=True)
            text = r.stdout + r.stderr
            lines += text.count("\n")
            name = docker("inspect", "-f", "{{.Name}}", cid).strip().lstrip("/")
            for c in self.report.canaries:
                if c in text:
                    hits.setdefault(name, []).append(c[:24])
        res.numbers.update(
            containers=len(ids),
            log_lines=lines,
            strings_searched=len(self.report.canaries),
            hits=hits,
        )
        check(not hits, f"prompt/completion/key text found in logs: {hits}")


def run(stack: Stack, only: str, report: Report) -> None:
    acc = Acceptance(stack, report)
    titles = {
        "0": "Sign up, approve, $5, key, SDK + playground, usage and balance",
        "a": "Usage = engine-reported tokens, incl. client abort and killed pod",
        "b": "Revoked key (and suspended org) refused within 5 s",
        "c": "Concurrent load cannot overshoot a small balance",
        "d": "Gateway p50 added overhead < 10 ms",
        "e": "Load test: TTFT and tokens/s through the gateway",
        "f": "Cold model: 503 model_unavailable + Retry-After, then ready",
        "g": "Stripe stub checkout credits once",
        "h": "Cross-tenant spot checks (web + API)",
        "i": "No prompt/completion/key text in container logs",
    }
    order = "0ahgbfcedi"
    wanted = set(only) | {"0"}
    try:
        for item in order:
            if item not in wanted:
                continue
            res = ItemResult(item, titles[item])
            say(f"\n== {item}: {res.title}")
            t0 = time.monotonic()
            try:
                getattr(acc, "flow" if item == "0" else f"item_{item}")(res)
            except Exception as e:  # recorded, and the run goes on
                res.status = "FAIL"
                res.notes.append(f"{type(e).__name__}: {e}")
            res.seconds = round(time.monotonic() - t0, 1)
            report.items.append(res)
            say(f"   {res.status} ({res.seconds} s) {json.dumps(res.numbers, default=str)}")
            for n in res.notes:
                say(f"   note: {n}")
            if item == "0" and res.status == "FAIL":
                raise AcceptanceError("the flow failed; the other items need its tenants")
    finally:
        acc.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="loom-m1-acceptance", description=__doc__.splitlines()[0])
    p.add_argument("--web")
    p.add_argument("--gateway", help="gateway base URL including /v1")
    p.add_argument("--mailpit")
    p.add_argument("--database-url", dest="database_url")
    p.add_argument("--redis-url", dest="redis_url")
    p.add_argument("--project", help="compose project name (default $COMPOSE_PROJECT_NAME or loom)")
    p.add_argument("--admin-email", dest="admin_email", default="admin@m1.test")
    p.add_argument("--only", default=ITEMS, help="items to run, e.g. 0abc (default all)")
    p.add_argument("--json", type=Path, help="write the results here")
    args = p.parse_args(argv)
    stack = stack_from_env(args)
    report = Report()
    say(f"M1 acceptance against {stack.web} / {stack.gateway} (project {stack.project})")
    with contextlib.suppress(AcceptanceError):
        run(stack, args.only.replace(",", ""), report)
    say("\n== summary")
    for i in report.items:
        say(f"{i.item}  {i.status:4}  {i.title}")
    if args.json:
        args.json.write_text(json.dumps(report.to_json(), indent=2, default=str))
    sys.exit(0 if report.passed and report.items else 1)


if __name__ == "__main__":
    main()
