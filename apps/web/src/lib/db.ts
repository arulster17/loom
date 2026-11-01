import "server-only";

import { Pool, type PoolClient } from "pg";

// One pool per server process. Kept on globalThis so dev hot reloads reuse it.
const g = globalThis as unknown as { __loomPool?: Pool };

export function databaseUrl(): string {
  return (
    process.env.DATABASE_URL ??
    process.env.LOOM_DATABASE_URL ??
    "postgresql://loom:loom@localhost:5432/loom_platform"
  );
}

export function getPool(): Pool {
  if (!g.__loomPool) {
    g.__loomPool = new Pool({
      connectionString: databaseUrl(),
      max: Number(process.env.LOOM_WEB_DB_POOL_MAX ?? 10),
      idleTimeoutMillis: 30_000,
    });
  }
  return g.__loomPool;
}

/** Closes the pool (tests). */
export async function closePool(): Promise<void> {
  const p = g.__loomPool;
  g.__loomPool = undefined;
  if (p) await p.end();
}

export type Db = Pick<PoolClient, "query">;

/** Runs `fn` in one transaction; rolls back on any error. */
export async function withTx<T>(fn: (c: PoolClient) => Promise<T>): Promise<T> {
  const client = await getPool().connect();
  try {
    await client.query("BEGIN");
    const out = await fn(client);
    await client.query("COMMIT");
    return out;
  } catch (e) {
    await client.query("ROLLBACK").catch(() => {});
    throw e;
  } finally {
    client.release();
  }
}
