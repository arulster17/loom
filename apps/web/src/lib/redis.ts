import "server-only";

import Redis from "ioredis";

import type { CounterStore } from "@/lib/signin-limit";

// One Redis client per server process (LOOM_REDIS_URL). Everything the web app does with
// Redis is optional: model status falls back to "unknown" and the sign-in rate limit
// fails open. So the client never queues or retries for long: short connect and command
// timeouts, no offline queue, so a down or frozen Redis costs a request at most ~1 s.

const g = globalThis as unknown as { __loomRedis?: Redis };

export function redis(): Redis {
  if (!g.__loomRedis) {
    g.__loomRedis = new Redis(process.env.LOOM_REDIS_URL ?? "redis://localhost:6379/0", {
      lazyConnect: true,
      maxRetriesPerRequest: 1,
      connectTimeout: 1000,
      commandTimeout: 1000,
      enableOfflineQueue: false,
    });
    g.__loomRedis.on("error", () => {}); // each caller handles its own failures
  }
  return g.__loomRedis;
}

/** The client, connected (first use, or after it gave up reconnecting). */
export async function connectedRedis(): Promise<Redis> {
  const r = redis();
  if (r.status === "wait" || r.status === "end") await r.connect();
  return r;
}

/** Fixed-window counters: INCR each key and (re)set its TTL, in one MULTI. */
export const redisCounterStore: CounterStore = {
  async incr(keys, ttlS) {
    const r = await connectedRedis();
    const tx = r.multi();
    for (const k of keys) tx.incr(k).expire(k, ttlS);
    const res = await tx.exec();
    if (!res) throw new Error("transaction aborted");
    return keys.map((_, i) => {
      const [err, n] = res[i * 2];
      if (err) throw err;
      return Number(n);
    });
  },
};
