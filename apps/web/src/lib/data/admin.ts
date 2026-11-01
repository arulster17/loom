import "server-only";

import type { PoolClient } from "pg";

import { assertPlatformAdmin, AuthzError, isUuid, type Actor, type OrgStatus } from "@/lib/authz";
import { getPool, withTx } from "@/lib/db";
import { audit } from "@/lib/data/orgs";

// Platform accounts (loom_platform.ledger.SYSTEM_ACCOUNTS).
export const REVENUE_ACCOUNT = "00000000-0000-0000-0000-000000000001";
export const PROMO_EXPENSE_ACCOUNT = "00000000-0000-0000-0000-000000000003";

export const MAX_ADJUSTMENT_MICROS = 10_000_000_000n; // $10,000 either way

/** Any admin write: re-reads the role inside the transaction, so a revoked admin whose
 * page is still open cannot act. */
async function lockAdmin(c: PoolClient, actor: Actor): Promise<void> {
  assertPlatformAdmin(actor);
  const r = await c.query<{ role: string }>("SELECT role FROM users WHERE id = $1 FOR SHARE", [actor.userId]);
  if (r.rows[0]?.role !== "admin") throw new AuthzError("Admins only.");
}

async function adminRead<T>(actor: Actor, fn: (c: PoolClient) => Promise<T>): Promise<T> {
  return withTx(async (c) => {
    await lockAdmin(c, actor);
    return fn(c);
  });
}

export interface OrgSummary {
  id: string;
  name: string;
  status: OrgStatus;
  ownerEmail: string | null;
  createdAt: Date;
  approvedAt: Date | null;
  suspendedReason: string | null;
  balanceMicros: bigint;
  activeKeys: number;
}

const ORG_SUMMARY_SQL = `
  SELECT o.id, o.name, o.status, o.created_at, o.approved_at, o.suspended_reason,
         (SELECT u.email FROM org_members m JOIN users u ON u.id = m.user_id
          WHERE m.org_id = o.id AND m.role = 'owner' ORDER BY m.created_at LIMIT 1) AS owner_email,
         loom_org_credit_total(o.id) - loom_org_usage_total(o.id) AS balance,
         (SELECT count(*) FROM api_keys k
          WHERE k.org_id = o.id AND k.revoked_at IS NULL AND NOT ('playground' = ANY (k.scopes))) AS active_keys
  FROM orgs o`;

type OrgSummaryRow = {
  id: string;
  name: string;
  status: OrgStatus;
  created_at: Date;
  approved_at: Date | null;
  suspended_reason: string | null;
  owner_email: string | null;
  balance: string;
  active_keys: string;
};

function toSummary(x: OrgSummaryRow): OrgSummary {
  return {
    id: x.id,
    name: x.name,
    status: x.status,
    ownerEmail: x.owner_email,
    createdAt: x.created_at,
    approvedAt: x.approved_at,
    suspendedReason: x.suspended_reason,
    balanceMicros: BigInt(x.balance),
    activeKeys: Number(x.active_keys),
  };
}

export async function listOrgs(actor: Actor, status?: OrgStatus): Promise<OrgSummary[]> {
  return adminRead(actor, async (c) => {
    const r = await c.query<OrgSummaryRow>(
      `${ORG_SUMMARY_SQL} WHERE ($1::text IS NULL OR o.status = $1) ORDER BY o.created_at DESC LIMIT 500`,
      [status ?? null],
    );
    return r.rows.map(toSummary);
  });
}

export interface AuditEntry {
  id: string;
  actorEmail: string | null;
  action: string;
  targetType: string;
  targetId: string;
  details: Record<string, unknown>;
  createdAt: Date;
}

export interface OrgDetail extends OrgSummary {
  members: { email: string | null; role: string }[];
  audit: AuditEntry[];
}

export async function getOrgDetail(actor: Actor, orgId: string): Promise<OrgDetail | null> {
  if (!isUuid(orgId)) return null;
  return adminRead(actor, async (c) => {
    const r = await c.query<OrgSummaryRow>(`${ORG_SUMMARY_SQL} WHERE o.id = $1`, [orgId]);
    if (!r.rows[0]) return null;
    const members = await c.query<{ email: string | null; role: string }>(
      `SELECT u.email, m.role FROM org_members m JOIN users u ON u.id = m.user_id
       WHERE m.org_id = $1 ORDER BY m.created_at`,
      [orgId],
    );
    return {
      ...toSummary(r.rows[0]),
      members: members.rows,
      audit: await auditRows(c, { targetType: "org", targetId: orgId, limit: 100 }),
    };
  });
}

async function auditRows(
  c: PoolClient,
  opts: { targetType?: string; targetId?: string; limit: number },
): Promise<AuditEntry[]> {
  const r = await c.query<{
    id: string;
    actor_email: string | null;
    action: string;
    target_type: string;
    target_id: string;
    details: Record<string, unknown>;
    created_at: Date;
  }>(
    `SELECT a.id, u.email AS actor_email, a.action, a.target_type, a.target_id, a.details, a.created_at
     FROM admin_audit_log a LEFT JOIN users u ON u.id = a.actor_user_id
     WHERE ($1::text IS NULL OR a.target_type = $1) AND ($2::text IS NULL OR a.target_id = $2)
     ORDER BY a.id DESC LIMIT $3`,
    [opts.targetType ?? null, opts.targetId ?? null, opts.limit],
  );
  return r.rows.map((x) => ({
    id: x.id,
    actorEmail: x.actor_email,
    action: x.action,
    targetType: x.target_type,
    targetId: x.target_id,
    details: x.details,
    createdAt: x.created_at,
  }));
}

export async function recentAudit(actor: Actor, limit = 50): Promise<AuditEntry[]> {
  return adminRead(actor, (c) => auditRows(c, { limit }));
}

function pgCode(e: unknown): string | undefined {
  return typeof e === "object" && e !== null && "code" in e ? String((e as { code: unknown }).code) : undefined;
}

/**
 * Approves a pending signup: loom_approve_org activates the org, posts the signup credit
 * (idempotency key promo:signup:<org>) and writes the audit row, in one transaction.
 */
export async function approveOrg(actor: Actor, orgId: string, promoMicros: bigint): Promise<void> {
  assertPlatformAdmin(actor);
  if (!isUuid(orgId)) throw new AuthzError("Org not found.", "not_found");
  try {
    await getPool().query("SELECT loom_approve_org($1, $2, $3)", [orgId, actor.userId, promoMicros.toString()]);
  } catch (e) {
    const code = pgCode(e);
    if (code === "42501") throw new AuthzError("Admins only.");
    if (code === "P0002") throw new AuthzError("Org not found.", "not_found");
    if (code === "55000") throw new AuthzError("This org is not waiting for approval.");
    throw e;
  }
}

/** Moves an org between statuses, with an audit row in the same transaction. */
async function transition(
  actor: Actor,
  orgId: string,
  from: OrgStatus,
  to: OrgStatus,
  action: string,
  reason: string | null,
): Promise<void> {
  if (!isUuid(orgId)) throw new AuthzError("Org not found.", "not_found");
  await withTx(async (c) => {
    await lockAdmin(c, actor);
    const r = await c.query<{ status: OrgStatus }>("SELECT status FROM orgs WHERE id = $1 FOR UPDATE", [orgId]);
    if (!r.rows[0]) throw new AuthzError("Org not found.", "not_found");
    if (r.rows[0].status !== from) {
      throw new AuthzError(`This org is ${r.rows[0].status.replace("_", " ")}, not ${from.replace("_", " ")}.`);
    }
    // The orgs trigger NOTIFYs loom_auth_changed on commit; the gateway drops the org's
    // cached key records, and its 4 s cache TTL bounds the delay anyway (contracts §6).
    await c.query(
      `UPDATE orgs SET status = $2::text,
              suspended_at = CASE WHEN $2::text = 'suspended' THEN now() ELSE NULL END,
              suspended_reason = CASE WHEN $2::text = 'suspended' THEN $3::text ELSE NULL END
       WHERE id = $1`,
      [orgId, to, reason],
    );
    await audit(c, actor.userId, action, "org", orgId, { from, to, ...(reason ? { reason } : {}) });
  });
}

export const rejectOrg = (actor: Actor, orgId: string, reason: string | null) =>
  transition(actor, orgId, "pending_approval", "rejected", "org.reject", reason);

/** Instant suspension: every key of the org is refused by the gateway within 5 s. */
export const suspendOrg = (actor: Actor, orgId: string, reason: string) =>
  transition(actor, orgId, "active", "suspended", "org.suspend", reason);

export const unsuspendOrg = (actor: Actor, orgId: string) =>
  transition(actor, orgId, "suspended", "active", "org.unsuspend", null);

/**
 * Adds (grant, against promo_expense) or removes (correction, against revenue) credit.
 * The audit row comes first so its id is the ledger idempotency key `admin:<audit id>`
 * (contracts §3), and both commit together.
 */
export async function adjustCredits(
  actor: Actor,
  orgId: string,
  amountMicros: bigint,
  memo: string,
): Promise<{ transactionId: string }> {
  if (!isUuid(orgId)) throw new AuthzError("Org not found.", "not_found");
  const note = memo.trim().slice(0, 500);
  if (!note) throw new RangeError("Say why: the note is required.");
  if (amountMicros === 0n) throw new RangeError("The amount cannot be zero.");
  if (amountMicros > MAX_ADJUSTMENT_MICROS || -amountMicros > MAX_ADJUSTMENT_MICROS) {
    throw new RangeError("Adjustments are at most $10,000 either way.");
  }
  return withTx(async (c) => {
    await lockAdmin(c, actor);
    const org = await c.query("SELECT 1 FROM orgs WHERE id = $1 FOR UPDATE", [orgId]);
    if (!org.rows[0]) throw new AuthzError("Org not found.", "not_found");
    const grant = amountMicros > 0n;
    const auditId = await audit(c, actor.userId, "org.credit_adjust", "org", orgId, {
      amount_micros: amountMicros.toString(),
      kind: grant ? "grant" : "correction",
      memo: note,
    });
    const r = await c.query<{ transaction_id: string }>(
      `SELECT transaction_id FROM loom_post_transaction(
         'adjustment', $1, $2, NULL, $3,
         ARRAY[loom_org_credit_account($2), $4::uuid], ARRAY[$5::bigint, $6::bigint])`,
      [
        `admin:${auditId}`,
        orgId,
        note,
        grant ? PROMO_EXPENSE_ACCOUNT : REVENUE_ACCOUNT,
        amountMicros.toString(),
        (-amountMicros).toString(),
      ],
    );
    return { transactionId: r.rows[0].transaction_id };
  });
}
