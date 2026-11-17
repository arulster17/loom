// Shared by the m3-*.test.ts regression tests (docs/m3-review-fixes-spec.md, section Web).
//
// - Auth.js's own HTTP endpoints, driven through @auth/core's `Auth` with the app's real
//   config (src/auth.config.ts), the way next-auth's route handlers (`handlers.GET/POST`
//   at /api/auth/*) reach them. next-auth itself cannot be imported outside the Next.js
//   bundler (it imports `next/server` unresolved), so `nextAuthSignIn` mirrors the
//   `signIn` server-side helper of next-auth 5 (node_modules/next-auth/lib/actions.js).
// - A tiny SMTP server that records every message, so a test sees whether a magic-link
//   email actually went out.

import net from "node:net";

import { Auth, createActionURL, raw, setEnvDefaults, skipCSRFCheck, type AuthConfig } from "@auth/core";
import { redirect } from "next/navigation";

import { authConfig } from "@/auth.config";
import { getPool } from "@/lib/db";

export const ORIGIN = "http://localhost:3000";

/**
 * The request headers `next/headers` answers with in a test that mocks it. Every
 * Auth.js request a `Browser` makes, and every server action a test calls, sets it, so
 * code may read the client's headers either from the Auth.js request or from Next.js.
 */
export const nextRequest = { headers: new Headers() };

/** The app's Auth.js config as `NextAuth(() => authConfig())` resolves it per request. */
export function resolvedConfig(): AuthConfig {
  const c = authConfig() as unknown as AuthConfig;
  c.secret ??= process.env.AUTH_SECRET;
  c.basePath ||= "/api/auth";
  setEnvDefaults(process.env, c, true);
  return c;
}

/** One browser: a cookie jar and a client IP (what Next.js puts in X-Forwarded-For). */
export class Browser {
  readonly cookies = new Map<string, string>();

  constructor(readonly ip: string = randomIp()) {}

  headers(extra: Record<string, string> = {}): Headers {
    const h = new Headers({
      host: "localhost:3000",
      "x-forwarded-for": this.ip,
      "x-forwarded-proto": "http",
      ...extra,
    });
    if (this.cookies.size) h.set("cookie", [...this.cookies].map(([k, v]) => `${k}=${v}`).join("; "));
    return h;
  }

  async request(method: "GET" | "POST", path: string, body?: Record<string, string>): Promise<Response> {
    const headers = this.headers(body ? { "content-type": "application/x-www-form-urlencoded" } : {});
    nextRequest.headers = headers;
    const req = new Request(`${ORIGIN}${path}`, {
      method,
      headers,
      body: body ? new URLSearchParams(body) : undefined,
    });
    const res = await Auth(req, resolvedConfig());
    for (const sc of res.headers.getSetCookie()) {
      const [pair, ...attrs] = sc.split(";");
      const i = pair.indexOf("=");
      const name = pair.slice(0, i).trim();
      const value = pair.slice(i + 1).trim();
      const expired = attrs.some((a) => /^\s*max-age=0\s*$/i.test(a)) || value === "";
      if (expired) this.cookies.delete(name);
      else this.cookies.set(name, value);
    }
    return res;
  }

  async csrfToken(): Promise<string> {
    const res = await this.request("GET", "/api/auth/csrf");
    return ((await res.json()) as { csrfToken: string }).csrfToken;
  }

  /** `POST /api/auth/signin/nodemailer`, as the browser posts Auth.js's own sign-in form. */
  async postMagicLink(email: string): Promise<Response> {
    const csrfToken = await this.csrfToken();
    return this.request("POST", "/api/auth/signin/nodemailer", {
      email,
      csrfToken,
      callbackUrl: `${ORIGIN}/dashboard`,
    });
  }

  /** `POST /api/auth/signout` with the CSRF token, as Auth.js's own sign-out page does. */
  async postSignOut(): Promise<Response> {
    const csrfToken = await this.csrfToken();
    return this.request("POST", "/api/auth/signout", { csrfToken, callbackUrl: `${ORIGIN}/signin` });
  }
}

/** Where Auth.js sent the browser. */
export function location(res: Response): string {
  return res.headers.get("location") ?? "";
}

/** Auth.js answered "email sent": the verify-request page (the config's check-email page),
 * not an error page. */
export function isCheckEmail(url: string): boolean {
  const u = new URL(url, ORIGIN);
  return !u.searchParams.has("error") && /\/(signin\/check-email|verify-request)$/.test(u.pathname);
}

export function randomIp(): string {
  const b = () => Math.floor(Math.random() * 254) + 1;
  return `10.${b()}.${b()}.${b()}`;
}

/**
 * next-auth 5's `signIn(provider, options)` as called from a server action (the copy in
 * node_modules/next-auth/lib/actions.js, minus the cookie jar): a POST to Auth.js with
 * `raw` and `skipCSRFCheck`, then `redirect()` to where Auth.js answered.
 */
export async function nextAuthSignIn(
  provider: string,
  options: Record<string, string | boolean | undefined> = {},
): Promise<string> {
  const headers = new Headers(nextRequest.headers);
  const { redirect: shouldRedirect = true, redirectTo, ...rest } = options;
  const callbackUrl = redirectTo?.toString() ?? headers.get("Referer") ?? "/";
  const config = resolvedConfig();
  const signInURL = createActionURL(
    "signin",
    headers.get("x-forwarded-proto") ?? "http",
    headers,
    process.env,
    config,
  );
  const url = `${signInURL}/${provider}?`;
  headers.set("Content-Type", "application/x-www-form-urlencoded");
  const body = new URLSearchParams({ ...(rest as Record<string, string>), callbackUrl });
  const res = (await Auth(new Request(url, { method: "POST", headers, body }), {
    ...config,
    raw,
    skipCSRFCheck,
  } as never)) as unknown as Response | { redirect?: string };
  const responseUrl = res instanceof Response ? res.headers.get("Location") : res.redirect;
  const redirectUrl = responseUrl ?? url;
  if (shouldRedirect) redirect(redirectUrl);
  return redirectUrl;
}

/** The URL of a Next.js `redirect()` thrown by a server action, else rethrows. */
export function redirectTarget(e: unknown): string {
  const digest = (e as { digest?: unknown } | null)?.digest;
  if (typeof digest === "string" && digest.startsWith("NEXT_REDIRECT")) return digest.split(";")[2];
  throw e;
}

export interface Mail {
  from: string;
  to: string[];
  data: string;
}

/** An SMTP server on 127.0.0.1 that accepts and records every message. */
export async function startSmtp(): Promise<{
  url: string;
  mails: Mail[];
  sentTo(addr: string): number;
  close(): Promise<void>;
}> {
  const mails: Mail[] = [];
  const sockets = new Set<net.Socket>();
  const server = net.createServer((sock) => {
    sockets.add(sock);
    sock.on("close", () => sockets.delete(sock));
    sock.on("error", () => {});
    let buf = "";
    let inData = false;
    let cur: Mail = { from: "", to: [], data: "" };
    const reply = (s: string) => sock.write(`${s}\r\n`);
    reply("220 localhost fake ESMTP");
    sock.on("data", (chunk) => {
      buf += chunk.toString("utf8");
      let i: number;
      while ((i = buf.indexOf("\r\n")) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 2);
        if (inData) {
          if (line === ".") {
            inData = false;
            mails.push(cur);
            cur = { from: "", to: [], data: "" };
            reply("250 OK queued");
          } else {
            cur.data += `${line.startsWith("..") ? line.slice(1) : line}\n`;
          }
          continue;
        }
        const cmd = line.slice(0, 4).toUpperCase();
        if (cmd === "EHLO") sock.write("250-localhost\r\n250 8BITMIME\r\n");
        else if (cmd === "MAIL") {
          cur.from = /<([^>]*)>/.exec(line)?.[1] ?? "";
          reply("250 OK");
        } else if (cmd === "RCPT") {
          cur.to.push((/<([^>]*)>/.exec(line)?.[1] ?? "").toLowerCase());
          reply("250 OK");
        } else if (cmd === "DATA") {
          inData = true;
          reply("354 End data with <CR><LF>.<CR><LF>");
        } else if (cmd === "QUIT") {
          reply("221 Bye");
          sock.end();
        } else reply("250 OK");
      }
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as net.AddressInfo;
  return {
    url: `smtp://127.0.0.1:${port}`,
    mails,
    sentTo: (addr) => mails.filter((m) => m.to.includes(addr.toLowerCase())).length,
    close: () =>
      new Promise<void>((resolve) => {
        for (const s of sockets) s.destroy();
        server.close(() => resolve());
      }),
  };
}

/** Waits until `check()` holds (nodemailer may still be closing its connection). */
export async function eventually(check: () => boolean, ms = 2000): Promise<void> {
  const end = Date.now() + ms;
  while (!check() && Date.now() < end) await new Promise((r) => setTimeout(r, 20));
}

export function form(fields: Record<string, string>): FormData {
  const f = new FormData();
  for (const [k, v] of Object.entries(fields)) f.set(k, v);
  return f;
}

/** Everything an admin write can change. Compared before and after a refusal. */
export async function adminSnapshot(): Promise<Record<string, unknown>> {
  const r = await getPool().query(
    `SELECT (SELECT string_agg(id || ':' || status, ',' ORDER BY id) FROM orgs) AS org_status,
            (SELECT string_agg(id || ':' || role, ',' ORDER BY id) FROM users) AS roles,
            (SELECT count(*) FROM ledger_transactions)::int AS ledger,
            (SELECT count(*) FROM ledger_entries)::int AS entries,
            (SELECT count(*) FROM admin_audit_log)::int AS audit`,
  );
  return r.rows[0];
}

/** The org's credit (ledger credit total), in micros. */
export async function creditMicros(orgId: string): Promise<bigint> {
  const r = await getPool().query<{ c: string }>("SELECT loom_org_credit_total($1) AS c", [orgId]);
  return BigInt(r.rows[0].c);
}
