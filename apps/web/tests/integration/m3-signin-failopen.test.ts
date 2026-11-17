// W1 (R11), docs/m3-review-fixes-spec.md: the magic-link limit still fails open when
// Redis is down, on both paths (Auth.js's own POST /api/auth/signin/nodemailer and the
// sign-in form). Redis points at a closed port; mail goes to a recording SMTP server.

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

import { email, hasDb } from "./helpers";
import { Browser, eventually, form, isCheckEmail, location, nextRequest, redirectTarget, startSmtp } from "./m3-helpers";

const saved = Object.fromEntries(
  ["LOOM_REDIS_URL", "LOOM_SIGNIN_EMAIL_LIMIT", "LOOM_SIGNIN_IP_LIMIT", "EMAIL_SERVER"].map((k) => [k, process.env[k]]),
);
process.env.LOOM_REDIS_URL = "redis://127.0.0.1:1/0"; // nothing listens on port 1
process.env.LOOM_SIGNIN_EMAIL_LIMIT = "1";
process.env.LOOM_SIGNIN_IP_LIMIT = "1";

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
vi.mock("next-auth", async () => ({ AuthError: (await import("@auth/core/errors")).AuthError }));

const { signInWithEmail } = await import("@/app/actions");

describe.skipIf(!hasDb)("W1: the magic-link limit fails open when Redis is down", () => {
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

  it("W1: sends every email, with a warning, through Auth.js's endpoint and the form", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const addr = email("w1-open");
      const b = new Browser();
      // Limits of 1 would refuse the second send; with Redis down nothing is refused.
      for (let i = 0; i < 3; i++) {
        const res = await b.postMagicLink(addr);
        expect(isCheckEmail(location(res)), location(res)).toBe(true);
      }
      nextRequest.headers = b.headers();
      let redirected = "";
      try {
        expect(await signInWithEmail(undefined, form({ email: addr }))).toBeUndefined();
      } catch (e) {
        redirected = redirectTarget(e);
      }
      expect(isCheckEmail(redirected), redirected).toBe(true);
      await eventually(() => smtp.sentTo(addr) >= 4);
      expect(smtp.sentTo(addr)).toBe(4);
      expect(warn).toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });
});
