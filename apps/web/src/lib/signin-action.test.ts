// signInWithEmail (src/app/actions.ts) with its rate limit: Auth.js, request headers
// and the Redis store are faked; the limiter itself is real.

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

const TOO_MANY = { error: "Too many sign-in emails. Try again in a few minutes." };

function send(email: string, ip = "203.0.113.1") {
  req.headers = new Headers({ "x-forwarded-for": `10.9.9.9, ${ip}` });
  const f = new FormData();
  f.set("email", email);
  return signInWithEmail(undefined, f);
}

describe("signInWithEmail rate limit", () => {
  beforeEach(() => {
    store.counts.clear();
    store.fail = null;
    vi.mocked(signIn).mockClear();
    for (const k of ["LOOM_SIGNIN_EMAIL_LIMIT", "LOOM_SIGNIN_IP_LIMIT", "LOOM_SIGNIN_WINDOW_S"]) delete process.env[k];
  });

  it("sends 5 emails to one address, then refuses without calling signIn", async () => {
    for (let i = 0; i < 5; i++) expect(await send("alice@example.com", `198.51.100.${i}`)).toBeUndefined();
    expect(signIn).toHaveBeenCalledTimes(5);
    expect(await send("  ALICE@example.com ", "198.51.100.200")).toEqual(TOO_MANY);
    expect(signIn).toHaveBeenCalledTimes(5);
  });

  it("refuses the 21st email from one client IP (the right-most forwarded entry)", async () => {
    for (let i = 0; i < 20; i++) expect(await send(`u${i}@example.com`, "198.51.100.7")).toBeUndefined();
    expect(await send("next@example.com", "198.51.100.7")).toEqual(TOO_MANY);
    expect(signIn).toHaveBeenCalledTimes(20);
    // A spoofed left-most entry does not move the client into a fresh bucket.
    req.headers = new Headers({ "x-forwarded-for": "1.2.3.4, 198.51.100.7" });
    const f = new FormData();
    f.set("email", "spoof@example.com");
    expect(await signInWithEmail(undefined, f)).toEqual(TOO_MANY);
  });

  it("reads the limits from the environment", async () => {
    process.env.LOOM_SIGNIN_EMAIL_LIMIT = "1";
    expect(await send("bob@example.com")).toBeUndefined();
    expect(await send("bob@example.com")).toEqual(TOO_MANY);
  });

  it("never stores the email in a Redis key name", async () => {
    await send("carol@example.com");
    expect([...store.counts.keys()].join(" ")).not.toMatch(/carol|example\.com|203\.0\.113\.1/);
  });

  it("fails open when Redis is down: the email is sent, with a warning", async () => {
    store.fail = new Error("connect ECONNREFUSED 127.0.0.1:6379");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(await send("dave@example.com")).toBeUndefined();
      expect(signIn).toHaveBeenCalledTimes(1);
      expect(warn).toHaveBeenCalledWith(expect.stringContaining("ECONNREFUSED"));
    } finally {
      warn.mockRestore();
    }
  });

  it("checks the address before counting it", async () => {
    expect(await send("not-an-email")).toEqual({ error: "Enter a valid email address." });
    expect(store.counts.size).toBe(0);
    expect(signIn).not.toHaveBeenCalled();
  });
});
