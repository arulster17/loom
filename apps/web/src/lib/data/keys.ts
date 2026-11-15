import "server-only";

import { generateApiKey } from "@/lib/apikeys";
import { AuthzError, canCreateKeys, canRevokeKey, isUuid, type OrgContext } from "@/lib/authz";
import { getPool, withTx, type Db } from "@/lib/db";
import { lockOrgStatus } from "@/lib/data/orgs";

/** Scope that marks the web app's own playground keys (src/lib/data/playground.ts). */
export const PLAYGROUND_SCOPE = "playground";

export const MAX_KEY_NAME = 100;

export interface ApiKeyView {
  id: string;
  name: string;
  prefix: string;
  createdAt: Date;
  createdBy: string | null;
  createdByEmail: string | null;
  lastUsedAt: Date | null;
  revokedAt: Date | null;
  spendLimitMicros: string | null;
  spentMicros: string;
}

/** The org's API keys, newest first. Playground keys are internal and not listed. */
export async function listKeys(ctx: OrgContext, db: Db = getPool()): Promise<ApiKeyView[]> {
  const r = await db.query<{
    id: string;
    name: string;
    prefix: string;
    created_at: Date;
    created_by: string | null;
    created_by_email: string | null;
    last_used_at: Date | null;
    revoked_at: Date | null;
    spend_limit_micros: string | null;
    spent_micros: string;
  }>(
    `SELECT k.id, k.name, k.prefix, k.created_at, k.created_by, u.email AS created_by_email,
            k.last_used_at, k.revoked_at, k.spend_limit_micros,
            loom_key_usage_total(k.id) AS spent_micros
     FROM api_keys k
     LEFT JOIN users u ON u.id = k.created_by
     WHERE k.org_id = $1 AND NOT ($2 = ANY (k.scopes))
     ORDER BY k.revoked_at IS NOT NULL, k.created_at DESC`,
    [ctx.orgId, PLAYGROUND_SCOPE],
  );
  return r.rows.map((x) => ({
    id: x.id,
    name: x.name,
    prefix: x.prefix,
    createdAt: x.created_at,
    createdBy: x.created_by,
    createdByEmail: x.created_by_email,
    lastUsedAt: x.last_used_at,
    revokedAt: x.revoked_at,
    spendLimitMicros: x.spend_limit_micros,
    spentMicros: x.spent_micros,
  }));
}

export interface CreateKeyInput {
  name: string;
  spendLimitMicros: bigint | null;
}

/**
 * Creates a key for the caller's org. Returns the plaintext, which is never stored: the
 * caller shows it once. Refused unless the org is active (re-checked under a row lock).
 */
export async function createKey(
  ctx: OrgContext,
  input: CreateKeyInput,
): Promise<{ id: string; prefix: string; plaintext: string }> {
  if (!canCreateKeys(ctx)) throw new AuthzError("Keys can be created once your account is approved.", "org_not_active");
  const name = input.name.trim().slice(0, MAX_KEY_NAME);
  if (input.spendLimitMicros !== null && input.spendLimitMicros < 0n) {
    throw new RangeError("Spend limit cannot be negative.");
  }
  return withTx(async (c) => {
    if ((await lockOrgStatus(c, ctx.orgId)) !== "active") {
      throw new AuthzError("This organization is not active.", "org_not_active");
    }
    const key = generateApiKey();
    const r = await c.query<{ id: string }>(
      `INSERT INTO api_keys (org_id, created_by, name, prefix, key_hash, spend_limit_micros)
       VALUES ($1, $2, $3, $4, $5, $6) RETURNING id`,
      [ctx.orgId, ctx.userId, name, key.prefix, key.keyHash, input.spendLimitMicros?.toString() ?? null],
    );
    return { id: r.rows[0].id, prefix: key.prefix, plaintext: key.plaintext };
  });
}

/**
 * Revokes one of the caller's org's keys. A key of another org is "not found", exactly
 * like a key that does not exist. The gateway refuses it within 5 s (contracts §6; the
 * api_keys trigger also notifies loom_auth_changed). Revoking twice is a no-op.
 */
export async function revokeKey(ctx: OrgContext, keyId: string): Promise<void> {
  if (!isUuid(keyId)) throw new AuthzError("Key not found.", "not_found");
  await withTx(async (c) => {
    const r = await c.query<{ created_by: string | null; revoked_at: Date | null }>(
      "SELECT created_by, revoked_at FROM api_keys WHERE id = $1 AND org_id = $2 FOR UPDATE",
      [keyId, ctx.orgId],
    );
    const key = r.rows[0];
    if (!key) throw new AuthzError("Key not found.", "not_found");
    if (!canRevokeKey(ctx, { createdBy: key.created_by })) {
      throw new AuthzError("Only the key's creator or an org owner or admin can revoke it.");
    }
    if (key.revoked_at) return;
    await c.query("UPDATE api_keys SET revoked_at = now() WHERE id = $1 AND org_id = $2", [keyId, ctx.orgId]);
  });
}
