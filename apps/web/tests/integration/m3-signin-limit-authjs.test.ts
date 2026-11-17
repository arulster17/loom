// W1 (R11), docs/m3-review-fixes-spec.md: the magic-link limit applies wherever Auth.js
// sends. Auth.js's own endpoint (POST /api/auth/signin/nodemailer, reached here through
// @auth/core with the app's real config) and the sign-in form (the signInWithEmail
// action, whose `signIn` is next-auth's, mirrored in m3-helpers.ts) share one per-email
// and per-IP budget, each send counted once. Mail goes to a recording SMTP server;
// counters to the real local Redis (database 15, or LOOM_TEST_REDIS_URL).

import Redis from "ioredis";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

import { email, hasDb } from "./helpers";
import { Browser, eventually, form, isCheckEmail, location, nextRequest, redirectTarget, startSmtp } from "./m3-helpers";

const redisUrl = process.env.LOOM_TEST_REDIS_URL ?? "redis://localhost:6379/15";
const saved = Object.fromEntries(
  ["LOOM_REDIS_URL", "LOOM_SIGNIN_EMAIL_LIMIT", "LOOM_SIGNIN_IP_LIMIT", "LOOM_SIGNIN_WINDOW_S", "EMAIL_SERVER"].map(
    (k) => [k, process.env[k]],
  ),
);
process.env.LOOM_REDIS_URL = redisUrl;
process.env.LOOM_SIGNIN_EMAIL_LIMIT = "2";
process.env.LOOM_SIGNIN_IP_LIMIT = "3";
// Fixed windows: a long one, so no test straddles a window boundary (the counts would reset).
process.env.LOOM_SIGNIN_WINDOW_S = "86400";

vi.mock("next/headers", async () => {
  const h = await import("./m3-helpers");
  return {
    headers: async () => h.nextRequest.headers,
    cookies: async () => ({ get: () => undefined, getAll: () => [], set: () => {}, has: () => false }),
  };
});
vi.mock("@/auth", async () => {
  const h = await import("./m3-helpers");
  return { signIn: h.nextAuthSignIn, signOut: vi.fn(async () => undefined), auth: vi.fn(async () => null) };
});
// next-auth re-exports @auth/core's errors; the package itself needs the Next.js bundler.
vi.mock("next-auth", async () => ({ AuthError: (await import("@auth/core/errors")).AuthError }));

const probe = new Redis(redisUrl, { lazyConnect: true, maxRetriesPerRequest: 0, connectTimeout: 1000, retryStrategy: () => null });
probe.on("error", () => {});
const redisUp = await probe.connect().then(
  () => true,
  () => false,
);
probe.disconnect();

const { signInWithEmail } = await import("@/app/actions");

const TOO_MANY = { error: "Too many sign-in emails. Try again in a few minutes." };

/** The sign-in form's submission, from its own browser (client IP). */
async function formSend(addr: string, b = new Browser()): Promise<{ error?: string; redirect?: string }> {
  nextRequest.headers = b.headers();
  try {
    return (await signInWithEmail(undefined, form({ email: addr }))) ?? {};
  } catch (e) {
    return { redirect: redirectTarget(e) };
  }
}

describe.skipIf(!hasDb || !redisUp)("W1: the magic-link limit applies wherever Auth.js sends", () => {
  let smtp: Awaited<ReturnType<typeof startSmtp>>;

  beforeAll(async () => {
    smtp = await startSmtp();
    process.env.EMAIL_SERVER = smtp.url;
  });

  afterAll(async () => {
    await smtp.close();
    for (const [k, v] of Object.entries(saved)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
    const { redis } = await import("@/lib/redis");
    redis().disconnect();
  });

  it("W1: limits a direct POST /api/auth/signin/nodemailer per email; over the limit no mail is sent", async () => {
    const addr = email("w1-direct");
    for (let i = 0; i < 2; i++) {
      const res = await new Browser().postMagicLink(addr);
      expect(isCheckEmail(location(res)), location(res)).toBe(true);
    }
    await eventually(() => smtp.sentTo(addr) >= 2);
    expect(smtp.sentTo(addr)).toBe(2);

    const over = await new Browser().postMagicLink(addr.toUpperCase());
    expect(isCheckEmail(location(over)), location(over)).toBe(false);
    await new Promise((r) => setTimeout(r, 200));
    expect(smtp.sentTo(addr)).toBe(2);
  });

  it("W1: limits a direct POST /api/auth/signin/nodemailer per client IP; over the limit no mail is sent", async () => {
    const b = new Browser();
    const addrs = [1, 2, 3, 4].map((i) => email(`w1-ip${i}`));
    for (const a of addrs.slice(0, 3)) {
      const res = await b.postMagicLink(a);
      expect(isCheckEmail(location(res)), location(res)).toBe(true);
    }
    const over = await b.postMagicLink(addrs[3]);
    expect(isCheckEmail(location(over)), location(over)).toBe(false);
    await eventually(() => addrs.slice(0, 3).every((a) => smtp.sentTo(a) === 1));
    await new Promise((r) => setTimeout(r, 200));
    expect(addrs.map((a) => smtp.sentTo(a))).toEqual([1, 1, 1, 0]);
  });

  it("W1: the form and Auth.js's endpoint share one budget, each send counted once", async () => {
    const addr = email("w1-shared");
    // Budget 2 per email: a form send and a direct POST both go out...
    const first = await formSend(addr);
    expect(first.error).toBeUndefined();
    expect(isCheckEmail(first.redirect ?? ""), first.redirect).toBe(true);
    const second = await new Browser().postMagicLink(addr);
    expect(isCheckEmail(location(second)), location(second)).toBe(true);
    await eventually(() => smtp.sentTo(addr) >= 2);
    // ...and then both paths are over the limit, and the form says so.
    expect(await formSend(addr)).toEqual(TOO_MANY);
    const fourth = await new Browser().postMagicLink(addr);
    expect(isCheckEmail(location(fourth)), location(fourth)).toBe(false);
    await new Promise((r) => setTimeout(r, 200));
    expect(smtp.sentTo(addr)).toBe(2);
  });

  it("W1: a send through the form counts once, and over the limit the form shows the message", async () => {
    const addr = email("w1-form");
    // Limit 2 per email: counted twice per send, the second would already be refused.
    for (let i = 0; i < 2; i++) {
      const r = await formSend(addr);
      expect(r.error).toBeUndefined();
      expect(isCheckEmail(r.redirect ?? ""), r.redirect).toBe(true);
    }
    expect(await formSend(addr)).toEqual(TOO_MANY);
    await new Promise((r) => setTimeout(r, 200));
    expect(smtp.sentTo(addr)).toBe(2);
  });

  it("W1: the form's per-IP limit is the same budget as Auth.js's endpoint", async () => {
    const b = new Browser();
    const addrs = [1, 2, 3, 4].map((i) => email(`w1-ipform${i}`));
    // IP budget 3: two direct POSTs and one form send from one client, then the form is refused.
    expect(isCheckEmail(location(await b.postMagicLink(addrs[0])))).toBe(true);
    expect(isCheckEmail(location(await b.postMagicLink(addrs[1])))).toBe(true);
    const third = await formSend(addrs[2], b);
    expect(isCheckEmail(third.redirect ?? ""), third.redirect).toBe(true);
    expect(await formSend(addrs[3], b)).toEqual(TOO_MANY);
    await new Promise((r) => setTimeout(r, 200));
    expect(smtp.sentTo(addrs[3])).toBe(0);
  });
});
