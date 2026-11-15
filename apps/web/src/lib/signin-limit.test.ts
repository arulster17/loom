import { describe, expect, it, vi } from "vitest";

import {
  checkSigninLimit,
  clientIp,
  DEFAULT_SIGNIN_LIMITS,
  normaliseEmail,
  signinLimitKeys,
  signinLimits,
  type CounterStore,
} from "@/lib/signin-limit";

/** In-memory fixed-window counters, like the Redis store (TTL ignored: keys carry the window). */
function memoryStore(): CounterStore & { counts: Map<string, number> } {
  const counts = new Map<string, number>();
  return {
    counts,
    async incr(keys) {
      return keys.map((k) => {
        const n = (counts.get(k) ?? 0) + 1;
        counts.set(k, n);
        return n;
      });
    },
  };
}

const SECRET = "test-secret";
const T0 = Date.UTC(2026, 10, 1, 12, 0, 0);

function check(store: CounterStore, email: string, ip: string, nowMs = T0, extra = {}) {
  return checkSigninLimit(store, { email, ip, nowMs, secret: SECRET, limits: DEFAULT_SIGNIN_LIMITS, ...extra });
}

describe("signinLimits", () => {
  it("defaults to 5 per email and 20 per IP per 15 minutes", () => {
    expect(signinLimits({})).toEqual({ emailLimit: 5, ipLimit: 20, windowS: 900 });
  });

  it("reads the env, ignoring anything that is not a positive integer", () => {
    expect(signinLimits({ LOOM_SIGNIN_EMAIL_LIMIT: "3", LOOM_SIGNIN_IP_LIMIT: "50", LOOM_SIGNIN_WINDOW_S: "60" })).toEqual({
      emailLimit: 3,
      ipLimit: 50,
      windowS: 60,
    });
    for (const bad of ["0", "-1", "1.5", "abc", "", " "]) {
      expect(signinLimits({ LOOM_SIGNIN_EMAIL_LIMIT: bad, LOOM_SIGNIN_IP_LIMIT: bad, LOOM_SIGNIN_WINDOW_S: bad })).toEqual(
        DEFAULT_SIGNIN_LIMITS,
      );
    }
  });
});

describe("clientIp", () => {
  const h = (fields: Record<string, string>) => new Headers(fields);

  it("takes the right-most X-Forwarded-For entry (the one our proxy appended)", () => {
    expect(clientIp(h({ "x-forwarded-for": "203.0.113.7" }))).toBe("203.0.113.7");
    expect(clientIp(h({ "x-forwarded-for": "1.1.1.1, 2.2.2.2,  198.51.100.9 " }))).toBe("198.51.100.9");
    expect(clientIp(h({ "x-forwarded-for": "spoofed, 2001:DB8::1" }))).toBe("2001:db8::1");
    expect(clientIp(h({ "x-forwarded-for": "1.1.1.1, , " }))).toBe("1.1.1.1");
  });

  it("falls back to X-Real-IP, then one shared unknown bucket", () => {
    expect(clientIp(h({ "x-real-ip": "192.0.2.1" }))).toBe("192.0.2.1");
    expect(clientIp(h({ "x-forwarded-for": " , ", "x-real-ip": "192.0.2.1" }))).toBe("192.0.2.1");
    expect(clientIp(h({}))).toBe("unknown");
  });
});

describe("signinLimitKeys", () => {
  it("never puts the email or the IP in the key name", () => {
    const keys = signinLimitKeys("Alice.Smith@Example.com", "203.0.113.7", 900, T0, SECRET);
    for (const k of Object.values(keys)) {
      expect(k.toLowerCase()).not.toContain("alice");
      expect(k).not.toContain("example");
      expect(k).not.toContain("203.0.113.7");
      expect(k).toMatch(/^loom:web:signin:(email|ip):[0-9a-f]{64}:\d+$/);
    }
  });

  it("normalises the email (trim, lowercase) before hashing", () => {
    expect(normaliseEmail("  Bob@Example.COM ")).toBe("bob@example.com");
    const a = signinLimitKeys(" Bob@Example.COM", "x", 900, T0, SECRET);
    const b = signinLimitKeys("bob@example.com", "x", 900, T0, SECRET);
    expect(a.email).toBe(b.email);
  });

  it("depends on the secret, so the hash cannot be checked against a list of emails without it", () => {
    expect(signinLimitKeys("a@b.c", "x", 900, T0, "one").email).not.toBe(signinLimitKeys("a@b.c", "x", 900, T0, "two").email);
  });
});

describe("checkSigninLimit", () => {
  it("allows 5 emails to one address in a window, then refuses", async () => {
    const store = memoryStore();
    for (let i = 0; i < 5; i++) {
      expect(await check(store, "a@example.com", `10.0.0.${i}`)).toEqual({ allowed: true });
    }
    expect(await check(store, "A@Example.com ", "10.0.0.99")).toEqual({ allowed: false, reason: "email" });
    // Another address is unaffected.
    expect(await check(store, "b@example.com", "10.0.0.99")).toEqual({ allowed: true });
  });

  it("allows 20 emails from one IP in a window, then refuses", async () => {
    const store = memoryStore();
    for (let i = 0; i < 20; i++) {
      expect((await check(store, `u${i}@example.com`, "198.51.100.1")).allowed).toBe(true);
    }
    expect(await check(store, "new@example.com", "198.51.100.1")).toEqual({ allowed: false, reason: "ip" });
    expect((await check(store, "new@example.com", "198.51.100.2")).allowed).toBe(true);
  });

  it("starts counting again in the next window", async () => {
    const store = memoryStore();
    for (let i = 0; i < 6; i++) await check(store, "a@example.com", "x");
    expect((await check(store, "a@example.com", "x")).allowed).toBe(false);
    expect((await check(store, "a@example.com", "x", T0 + 900_000)).allowed).toBe(true);
  });

  it("uses the configured limits and window", async () => {
    const store = memoryStore();
    const limits = { emailLimit: 1, ipLimit: 100, windowS: 60 };
    expect((await check(store, "a@example.com", "x", T0, { limits })).allowed).toBe(true);
    expect((await check(store, "a@example.com", "x", T0, { limits })).allowed).toBe(false);
    expect((await check(store, "a@example.com", "x", T0 + 60_000, { limits })).allowed).toBe(true);
  });

  it("asks the store to expire the keys after one window", async () => {
    const incr = vi.fn(async (keys: string[]) => keys.map(() => 1));
    await check({ incr }, "a@example.com", "x", T0, { limits: { emailLimit: 5, ipLimit: 20, windowS: 120 } });
    expect(incr).toHaveBeenCalledWith([expect.stringContaining(":email:"), expect.stringContaining(":ip:")], 120);
  });

  it("fails open, with a warning, when the store errors", async () => {
    const warn = vi.fn();
    const store: CounterStore = { incr: async () => Promise.reject(new Error("ECONNREFUSED")) };
    expect(await check(store, "a@example.com", "x", T0, { warn })).toEqual({ allowed: true, failedOpen: true });
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("ECONNREFUSED"));
    expect(warn.mock.calls[0][0]).not.toContain("a@example.com");
  });

  it("fails open when the store does not answer in time (a frozen Redis cannot hang sign-in)", async () => {
    const warn = vi.fn();
    const store: CounterStore = { incr: () => new Promise(() => {}) };
    const started = Date.now();
    expect(await check(store, "a@example.com", "x", T0, { warn, timeoutMs: 50 })).toEqual({ allowed: true, failedOpen: true });
    expect(Date.now() - started).toBeLessThan(1000);
    expect(warn).toHaveBeenCalledOnce();
  });
});
