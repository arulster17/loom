import { randomUUID } from "node:crypto";

import PostgresAdapter from "@auth/pg-adapter";
import { inject } from "vitest";

import type { Actor, OrgContext, OrgRole } from "@/lib/authz";
import { getPool } from "@/lib/db";
import { approveOrg } from "@/lib/data/admin";
import { ensureUserOrg, loadActor, loadOrgContext } from "@/lib/data/orgs";

export const hasDb = !!inject("databaseUrl");

export function email(tag = "user"): string {
  return `${tag}-${randomUUID().slice(0, 8)}@example.com`;
}

export function adapter() {
  return PostgresAdapter(getPool());
}

/** A user signed up through the Auth.js adapter, with their (pending) org. */
export async function signUp(addr = email()): Promise<OrgContext> {
  const user = await adapter().createUser!({ id: randomUUID(), email: addr, emailVerified: new Date(), name: null, image: null });
  await ensureUserOrg(user.id);
  return (await loadOrgContext(user.id))!;
}

export async function makeAdmin(): Promise<Actor> {
  const r = await getPool().query<{ id: string }>(
    "INSERT INTO users (email, role) VALUES ($1, 'admin') RETURNING id",
    [email("admin")],
  );
  return (await loadActor(r.rows[0].id))!;
}

/** A signed-up user whose org an admin approved ($5 credit posted). */
export async function activeUser(admin: Actor, addr = email()): Promise<OrgContext> {
  const ctx = await signUp(addr);
  await approveOrg(admin, ctx.orgId, 5_000_000n);
  return (await loadOrgContext(ctx.userId))!;
}

/** Another user added to an existing org with the given role. */
export async function addMember(orgId: string, role: OrgRole): Promise<OrgContext> {
  const r = await getPool().query<{ id: string }>("INSERT INTO users (email) VALUES ($1) RETURNING id", [email(role)]);
  await getPool().query("INSERT INTO org_members (org_id, user_id, role) VALUES ($1, $2, $3)", [orgId, r.rows[0].id, role]);
  return (await loadOrgContext(r.rows[0].id))!;
}

export async function insertUsage(
  ctx: Pick<OrgContext, "orgId">,
  apiKeyId: string,
  opts: { costMicros?: number; model?: string; at?: Date; prompt?: number; completion?: number } = {},
): Promise<void> {
  await getPool().query(
    `INSERT INTO usage_events (id, org_id, api_key_id, model_id, price_version, cloud, prompt_tokens,
       completion_tokens, latency_ms, status, cost_micros, created_at)
     VALUES ($1, $2, $3, $4, '77000:77000:368000', 'local', $5, $6, 100, 'ok', $7, $8)`,
    [
      randomUUID(),
      ctx.orgId,
      apiKeyId,
      opts.model ?? "qwen3-8b",
      opts.prompt ?? 100,
      opts.completion ?? 50,
      opts.costMicros ?? 1000,
      opts.at ?? new Date(),
    ],
  );
}

export async function keyIdByPrefix(prefix: string): Promise<string> {
  const r = await getPool().query<{ id: string }>("SELECT id FROM api_keys WHERE prefix = $1", [prefix]);
  return r.rows[0].id;
}
