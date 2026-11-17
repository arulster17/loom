// signInWithEmail (src/app/actions.ts): Auth.js, request headers and the Redis store are
// faked. The magic-link limit is checked where Auth.js sends (docs/m3-review-fixes-spec.md
// W1, tests/integration/m3-signin-*.test.ts); the limiter's own rules are in
// signin-limit.test.ts. Here: the action validates the address and does not count it itself.

import { beforeEach, describe, expect, it, vi } from "vitest";

import type { CounterStore } from "@/lib/signin-limit";

const req = vi.hoisted(() => ({ headers: new Headers() }));
const store = vi.hoisted(() => ({
  counts: new Map<string, number>(),
  fail: null as Error | null,
}));

vi.mock("next-auth", () => ({ AuthError: class AuthError extends Error {} }));
vi.mock("next/headers", () => ({ headers: vi.fn(async () => req.headers) }));
vi.mock("@/auth", () => ({ signIn: vi.fn(async () => undefined), signOut: vi.fn() }));
vi.mock("@/lib/session", () => ({ sessionUserId: vi.fn() }));
vi.mock("@/lib/data/playground", () => ({ revokePlaygroundKeys: vi.fn() }));
vi.mock("@/lib/redis", () => ({
  redisCounterStore: {
    async incr(keys: string[]) {
      if (store.fail) throw store.fail;
      return keys.map((k) => {
        const n = (store.counts.get(k) ?? 0) + 1;
        store.counts.set(k, n);
        return n;
      });
    },
  } satisfies CounterStore,
}));

const { signIn } = await import("@/auth");
const { signInWithEmail } = await import("@/app/actions");

function send(email: string, ip = "203.0.113.1") {
  req.headers = new Headers({ "x-forwarded-for": `10.9.9.9, ${ip}` });
  const f = new FormData();
  f.set("email", email);
  return signInWithEmail(undefined, f);
}


describe("signInWithEmail", () => {
  beforeEach(() => {
    store.counts.clear();
    store.fail = null;
    vi.mocked(signIn).mockClear();
    for (const k of ["LOOM_SIGNIN_EMAIL_LIMIT", "LOOM_SIGNIN_IP_LIMIT", "LOOM_SIGNIN_WINDOW_S"]) delete process.env[k];
  });

  it("passes a valid address to Auth.js and leaves the counting to Auth.js's send", async () => {
    process.env.LOOM_SIGNIN_EMAIL_LIMIT = "1";
    for (let i = 0; i < 3; i++) expect(await send("alice@example.com")).toBeUndefined();
    expect(signIn).toHaveBeenCalledTimes(3);
    expect(vi.mocked(signIn).mock.calls[0][0]).toBe("nodemailer");
    expect(store.counts.size).toBe(0);
  });

  it("checks the address before sending", async () => {
    expect(await send("not-an-email")).toEqual({ error: "Enter a valid email address." });
    expect(store.counts.size).toBe(0);
    expect(signIn).not.toHaveBeenCalled();
  });
});
