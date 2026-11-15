import "server-only";

// The playground's server-side, session-scoped key.
//
// The browser never holds an API key. It calls this app's /api/playground/* routes with
// its session cookie; the route checks the session and org, then calls the gateway with
// a short-lived "playground key" that exists only on the server:
//
// - One per (org, user), an ordinary api_keys row with scopes = {playground} and name
//   "Playground", so the gateway needs nothing new: limits, credit, suspension and usage
//   all work as for any key, and usage shows up as "Playground".
// - Its plaintext is never stored. The secret half is HMAC-derived from a server secret
//   and the stored prefix (src/lib/playground-key.ts), so the server can rebuild it on
//   each request while the database still holds only prefix + SHA-256.
// - It lives LOOM_PLAYGROUND_KEY_TTL_S (default 1 h): an older one is revoked and a new
//   one minted. Signing out revokes the user's playground keys at once.
// - Hidden from the API keys page; it cannot be copied or revoked there.

import { assembleApiKey, KEY_TAG, randomBase62 } from "@/lib/apikeys";
import { assertOrgActive, AuthzError, type OrgContext } from "@/lib/authz";
import { getPool, withTx } from "@/lib/db";
import { PLAYGROUND_SCOPE } from "@/lib/data/keys";
import { lockOrgStatus } from "@/lib/data/orgs";
import { derivePlaygroundSecret, playgroundServerSecret } from "@/lib/playground-key";

export function playgroundKeyTtlS(): number {
  const v = Number(process.env.LOOM_PLAYGROUND_KEY_TTL_S ?? 3600);
  return Number.isFinite(v) && v >= 600 ? Math.floor(v) : 3600;
}

/** A key is reused only while it has at least this long left. */
const MIN_REMAINING_S = 300;

/** The caller's playground key (plaintext, server-side only), minting one if needed. */
export async function getPlaygroundKey(ctx: OrgContext): Promise<string> {
  assertOrgActive(ctx);
  const secret = playgroundServerSecret();
  const ttl = playgroundKeyTtlS();
  return withTx(async (c) => {
    if ((await lockOrgStatus(c, ctx.orgId)) !== "active") {
      throw new AuthzError("This organization is not active.", "org_not_active");
    }
    await c.query("SELECT pg_advisory_xact_lock(hashtext('loom-web:playground:' || $1 || ':' || $2))", [
      ctx.orgId,
      ctx.userId,
    ]);
    // Expire old playground keys (any org's: they are this app's own internal keys).
    await c.query(
      `UPDATE api_keys SET revoked_at = now()
       WHERE $1 = ANY (scopes) AND revoked_at IS NULL AND created_at < now() - make_interval(secs => $2)`,
      [PLAYGROUND_SCOPE, ttl],
    );
    const live = await c.query<{ prefix: string }>(
      `SELECT prefix FROM api_keys
       WHERE org_id = $1 AND created_by = $2 AND $3 = ANY (scopes) AND revoked_at IS NULL
         AND created_at > now() - make_interval(secs => $4)
       ORDER BY created_at DESC LIMIT 1`,
      [ctx.orgId, ctx.userId, PLAYGROUND_SCOPE, ttl - MIN_REMAINING_S],
    );
    if (live.rows[0]) {
      const prefix = live.rows[0].prefix;
      return assembleApiKey(prefix, derivePlaygroundSecret(secret, prefix)).plaintext;
    }
    const prefix = KEY_TAG + randomBase62(12);
    const key = assembleApiKey(prefix, derivePlaygroundSecret(secret, prefix));
    await c.query(
      `INSERT INTO api_keys (org_id, created_by, name, prefix, key_hash, scopes)
       VALUES ($1, $2, 'Playground', $3, $4, ARRAY[$5])`,
      [ctx.orgId, ctx.userId, key.prefix, key.keyHash, PLAYGROUND_SCOPE],
    );
    return key.plaintext;
  });
}

/** Sign-out: the user's playground keys stop working (gateway refuses within 5 s). */
export async function revokePlaygroundKeys(userId: string): Promise<number> {
  const r = await getPool().query(
    `UPDATE api_keys SET revoked_at = now()
     WHERE created_by = $1 AND $2 = ANY (scopes) AND revoked_at IS NULL`,
    [userId, PLAYGROUND_SCOPE],
  );
  return r.rowCount ?? 0;
}
