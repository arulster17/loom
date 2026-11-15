import "server-only";

import type { PoolClient } from "pg";

import { isBootstrapAdmin, type Actor, type OrgContext, type OrgRole, type OrgStatus, type UserRole } from "@/lib/authz";
import { getPool, withTx, type Db } from "@/lib/db";

/** "alice.smith@example.com" -> "alice.smith". Users can be shown a nicer name later. */
export function defaultOrgName(email: string | null | undefined): string {
  const local = (email ?? "").split("@")[0]?.trim();
  return local ? `${local}'s org` : "My org";
}

/**
 * Gives a new user their own org (pending approval) with them as owner, in one
 * transaction (contracts §4). Idempotent: a user who already belongs to an org keeps it,
 * and an advisory lock on the user id stops two concurrent sign-ins creating two orgs.
 * Returns the org id.
 */
export async function ensureUserOrg(userId: string): Promise<string> {
  return withTx(async (c) => {
    await c.query("SELECT pg_advisory_xact_lock(hashtext('loom-web:user-org:' || $1))", [userId]);
    const existing = await c.query<{ org_id: string }>(
      "SELECT org_id FROM org_members WHERE user_id = $1 ORDER BY created_at, org_id LIMIT 1",
      [userId],
    );
    if (existing.rows[0]) return existing.rows[0].org_id;
    const user = await c.query<{ email: string | null }>("SELECT email FROM users WHERE id = $1", [userId]);
    if (!user.rows[0]) throw new Error("user does not exist");
    const org = await c.query<{ id: string }>(
      "INSERT INTO orgs (name) VALUES ($1) RETURNING id",
      [defaultOrgName(user.rows[0].email)],
    );
    const orgId = org.rows[0].id;
    await c.query("INSERT INTO org_members (org_id, user_id, role) VALUES ($1, $2, 'owner')", [
      orgId,
      userId,
    ]);
    return orgId;
  });
}

/**
 * Local dev bootstrap (contracts §4): a user whose email is in LOOM_ADMIN_EMAILS gets
 * users.role = 'admin' at sign-in. The grant is audited in the same transaction.
 * Returns true if this call granted it.
 */
export async function syncBootstrapAdmin(
  userId: string,
  adminEmailsRaw: string | undefined = process.env.LOOM_ADMIN_EMAILS,
): Promise<boolean> {
  return withTx(async (c) => {
    const r = await c.query<{ email: string | null; role: UserRole }>(
      "SELECT email, role FROM users WHERE id = $1 FOR UPDATE",
      [userId],
    );
    const user = r.rows[0];
    if (!user || user.role === "admin" || !isBootstrapAdmin(user.email, adminEmailsRaw)) {
      return false;
    }
    await c.query("UPDATE users SET role = 'admin' WHERE id = $1", [userId]);
    await audit(c, userId, "user.grant_admin", "user", userId, { via: "LOOM_ADMIN_EMAILS" });
    return true;
  });
}

export async function audit(
  c: Db,
  actorUserId: string,
  action: string,
  targetType: string,
  targetId: string,
  details: Record<string, unknown> = {},
): Promise<string> {
  const r = await c.query<{ id: string }>(
    `INSERT INTO admin_audit_log (actor_user_id, action, target_type, target_id, details)
     VALUES ($1, $2, $3, $4, $5) RETURNING id`,
    [actorUserId, action, targetType, targetId, JSON.stringify(details)],
  );
  return r.rows[0].id;
}

export async function loadActor(userId: string, db: Db = getPool()): Promise<Actor | null> {
  const r = await db.query<{ id: string; email: string | null; role: UserRole }>(
    "SELECT id, email, role FROM users WHERE id = $1",
    [userId],
  );
  const u = r.rows[0];
  return u ? { userId: u.id, email: u.email ?? "", userRole: u.role } : null;
}

/**
 * The caller's context: who they are and the org they act in (their oldest membership;
 * M1 gives every user exactly one org). Always read fresh from the database, so a
 * suspension or role change applies on the next request.
 */
export async function loadOrgContext(userId: string, db: Db = getPool()): Promise<OrgContext | null> {
  const r = await db.query<{
    user_id: string;
    email: string | null;
    user_role: UserRole;
    org_id: string;
    org_name: string;
    org_role: OrgRole;
    org_status: OrgStatus;
  }>(
    `SELECT u.id AS user_id, u.email, u.role AS user_role,
            o.id AS org_id, o.name AS org_name, m.role AS org_role, o.status AS org_status
     FROM users u
     JOIN org_members m ON m.user_id = u.id
     JOIN orgs o ON o.id = m.org_id
     WHERE u.id = $1
     ORDER BY m.created_at, m.org_id
     LIMIT 1`,
    [userId],
  );
  const x = r.rows[0];
  if (!x) return null;
  return {
    userId: x.user_id,
    email: x.email ?? "",
    userRole: x.user_role,
    orgId: x.org_id,
    orgName: x.org_name,
    orgRole: x.org_role,
    orgStatus: x.org_status,
  };
}

/** Re-reads the org's status inside a transaction, locking the row against a concurrent
 * suspension, so a write never relies on a status read earlier in the request. */
export async function lockOrgStatus(c: PoolClient, orgId: string): Promise<OrgStatus | null> {
  const r = await c.query<{ status: OrgStatus }>("SELECT status FROM orgs WHERE id = $1 FOR SHARE", [orgId]);
  return r.rows[0]?.status ?? null;
}
