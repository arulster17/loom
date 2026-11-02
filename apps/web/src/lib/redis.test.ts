// The real ioredis client against a Redis that is down or frozen: the sign-in limit must
// fail open quickly, never hang. (No Redis needed: a closed port and a TCP server that
// accepts connections and never answers.)

import { createServer, type Server, type Socket } from "node:net";

import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import { checkSigninLimit } from "@/lib/signin-limit";

let frozen: Server;
let frozenPort: number;
const sockets: Socket[] = [];

beforeAll(async () => {
  frozen = createServer((s) => {
    sockets.push(s); // read everything, answer nothing
    s.on("error", () => {});
  });
  await new Promise<void>((resolve) => frozen.listen(0, "127.0.0.1", resolve));
  frozenPort = (frozen.address() as { port: number }).port;
});

afterAll(async () => {
  for (const s of sockets) s.destroy();
  await new Promise((resolve) => frozen.close(resolve));
});

async function freshClient(url: string) {
  const g = globalThis as unknown as { __loomRedis?: { disconnect(): void } };
  g.__loomRedis?.disconnect();
  delete g.__loomRedis;
  process.env.LOOM_REDIS_URL = url;
  return import("@/lib/redis");
}

afterEach(() => {
  const g = globalThis as unknown as { __loomRedis?: { disconnect(): void } };
  g.__loomRedis?.disconnect();
  delete g.__loomRedis;
  delete process.env.LOOM_REDIS_URL;
});

describe("redisCounterStore when Redis is unavailable", () => {
  it.each([
    ["frozen (accepts, never answers)", () => `redis://127.0.0.1:${frozenPort}/0`],
    ["down (connection refused)", () => "redis://127.0.0.1:1/0"],
  ])("%s: the sign-in limit fails open within ~1.5 s", async (_label, url) => {
    const { redisCounterStore } = await freshClient(url());
    const warn = vi.fn();
    const started = Date.now();
    const verdict = await checkSigninLimit(redisCounterStore, { email: "a@example.com", ip: "x", secret: "s", warn });
    expect(verdict).toEqual({ allowed: true, failedOpen: true });
    expect(Date.now() - started).toBeLessThan(2500);
    expect(warn).toHaveBeenCalledOnce();
  });
});
