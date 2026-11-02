// The sign-in rate limit's Redis store against the real local Redis (docker compose up
// -d redis). Uses database 15 (LOOM_TEST_REDIS_URL to change) and only keys under a
// random secret, deleted at the end. Skips when Redis is not reachable.

import { randomUUID } from "node:crypto";

import Redis from "ioredis";
import { afterAll, describe, expect, it } from "vitest";

import { checkSigninLimit, signinLimitKeys } from "@/lib/signin-limit";

const url = process.env.LOOM_TEST_REDIS_URL ?? "redis://localhost:6379/15";
process.env.LOOM_REDIS_URL = url;

const probe = new Redis(url, { lazyConnect: true, maxRetriesPerRequest: 0, connectTimeout: 1000, retryStrategy: () => null });
probe.on("error", () => {});
const up = await probe.connect().then(
  () => true,
  () => false,
);

const { redisCounterStore, redis } = await import("@/lib/redis");

describe.skipIf(!up)("sign-in limit on real Redis", () => {
  const secret = randomUUID();
  const limits = { emailLimit: 2, ipLimit: 3, windowS: 30 };
  const now = Date.now();

  afterAll(async () => {
    const mine = ["a@example.com", "b@example.com"].flatMap((e) =>
      Object.values(signinLimitKeys(e, "ip-1", limits.windowS, now, secret)),
    );
    await probe.del(...mine);
    probe.disconnect();
    redis().disconnect();
  });

  it("counts per email and per IP, with a TTL of one window", async () => {
    const run = (email: string) => checkSigninLimit(redisCounterStore, { email, ip: "ip-1", secret, limits, nowMs: now });
    expect(await run("a@example.com")).toEqual({ allowed: true });
    expect(await run("A@example.com")).toEqual({ allowed: true });
    expect(await run("a@example.com")).toEqual({ allowed: false, reason: "email" });
    expect(await run("b@example.com")).toEqual({ allowed: false, reason: "ip" }); // 4th from ip-1

    const keys = signinLimitKeys("a@example.com", "ip-1", limits.windowS, now, secret);
    expect(await probe.get(keys.email)).toBe("3");
    expect(await probe.get(keys.ip)).toBe("4");
    const ttl = await probe.ttl(keys.email);
    expect(ttl).toBeGreaterThan(0);
    expect(ttl).toBeLessThanOrEqual(limits.windowS);
  });
});
