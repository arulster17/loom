import "server-only";

import { AuthzError, canManageBilling, type OrgContext } from "@/lib/authz";
import { getPool, withTx, type Db } from "@/lib/db";
import { lockOrgStatus } from "@/lib/data/orgs";

/** Purchase and auto-recharge amounts, micro-dollars. */
export const MIN_TOPUP_MICROS = 5_000_000n; // $5
export const MAX_TOPUP_MICROS = 1_000_000_000n; // $1,000

export interface Balance {
  creditMicros: bigint; // purchases, promos, refunds, adjustments
  usageMicros: bigint; // every metered request, batched to the ledger or not
  balanceMicros: bigint; // credit - usage (contracts §3)
}

export async function getBalance(ctx: Pick<OrgContext, "orgId">, db: Db = getPool()): Promise<Balance> {
  const r = await db.query<{ credit: string; usage: string }>(
    "SELECT loom_org_credit_total($1) AS credit, loom_org_usage_total($1) AS usage",
    [ctx.orgId],
  );
  const credit = BigInt(r.rows[0].credit);
  const usage = BigInt(r.rows[0].usage);
  return { creditMicros: credit, usageMicros: usage, balanceMicros: credit - usage };
}

export interface CreditEntry {
  id: string;
  kind: "purchase" | "promo" | "refund" | "adjustment";
  amountMicros: bigint; // + adds credit, - removes it
  externalRef: string | null;
  memo: string | null;
  createdAt: Date;
}

/** Every non-usage ledger transaction of the org (the receipts list), newest first. */
export async function creditHistory(ctx: Pick<OrgContext, "orgId">, limit = 100, db: Db = getPool()): Promise<CreditEntry[]> {
  const r = await db.query<{
    id: string;
    kind: CreditEntry["kind"];
    amount: string;
    external_ref: string | null;
    memo: string | null;
    created_at: Date;
  }>(
    `SELECT t.id, t.kind, e.amount_micros AS amount, t.external_ref, t.memo, t.created_at
     FROM ledger_transactions t
     JOIN ledger_entries e ON e.transaction_id = t.id
     WHERE t.org_id = $1 AND t.kind <> 'usage' AND e.account_id = loom_org_credit_account($1)
     ORDER BY t.created_at DESC
     LIMIT $2`,
    [ctx.orgId, limit],
  );
  return r.rows.map((x) => ({
    id: x.id,
    kind: x.kind,
    amountMicros: BigInt(x.amount),
    externalRef: x.external_ref,
    memo: x.memo,
    createdAt: x.created_at,
  }));
}

export interface AutoRecharge {
  enabled: boolean;
  thresholdMicros: bigint;
  amountMicros: bigint;
}

export const DEFAULT_AUTO_RECHARGE: AutoRecharge = {
  enabled: false,
  thresholdMicros: 5_000_000n,
  amountMicros: 20_000_000n,
};

export async function getAutoRecharge(ctx: Pick<OrgContext, "orgId">, db: Db = getPool()): Promise<AutoRecharge> {
  const r = await db.query<{ enabled: boolean; threshold_micros: string; amount_micros: string }>(
    "SELECT enabled, threshold_micros, amount_micros FROM auto_recharge WHERE org_id = $1",
    [ctx.orgId],
  );
  const x = r.rows[0];
  if (!x) return DEFAULT_AUTO_RECHARGE;
  return { enabled: x.enabled, thresholdMicros: BigInt(x.threshold_micros), amountMicros: BigInt(x.amount_micros) };
}

export function validateTopup(amountMicros: bigint): void {
  if (amountMicros < MIN_TOPUP_MICROS || amountMicros > MAX_TOPUP_MICROS) {
    throw new RangeError("Amounts are between $5 and $1,000.");
  }
}

/**
 * Saves the org's auto-recharge settings. The billing service acts on them (charging
 * the saved card when the balance drops below the threshold); this only stores them.
 */
export async function saveAutoRecharge(ctx: OrgContext, s: AutoRecharge): Promise<void> {
  if (!canManageBilling(ctx)) throw new AuthzError("Only org owners and admins can change billing.");
  validateTopup(s.amountMicros);
  if (s.thresholdMicros < 0n || s.thresholdMicros > MAX_TOPUP_MICROS) {
    throw new RangeError("The threshold is between $0 and $1,000.");
  }
  await withTx(async (c) => {
    if ((await lockOrgStatus(c, ctx.orgId)) !== "active") {
      throw new AuthzError("This organization is not active.", "org_not_active");
    }
    await c.query(
      `INSERT INTO auto_recharge (org_id, enabled, threshold_micros, amount_micros)
       VALUES ($1, $2, $3, $4)
       ON CONFLICT (org_id) DO UPDATE
       SET enabled = EXCLUDED.enabled, threshold_micros = EXCLUDED.threshold_micros,
           amount_micros = EXCLUDED.amount_micros`,
      [ctx.orgId, s.enabled, s.thresholdMicros.toString(), s.amountMicros.toString()],
    );
  });
}
