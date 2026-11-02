import { Client } from "pg";
import { beforeAll, describe, expect, it } from "vitest";

import { AuthzError, type Actor } from "@/lib/authz";
import { getPool } from "@/lib/db";
import {
  adjustCredits,
  approveOrg,
  getOrgDetail,
  listOrgs,
  recentAudit,
  rejectOrg,
  suspendOrg,
  unsuspendOrg,
} from "@/lib/data/admin";
import { getBalance } from "@/lib/data/billing";
import { loadOrgContext } from "@/lib/data/orgs";

import { activeUser, hasDb, makeAdmin, signUp } from "./helpers";

async function auditActions(orgId: string): Promise<string[]> {
  const r = await getPool().query("SELECT action FROM admin_audit_log WHERE target_type = 'org' AND target_id = $1 ORDER BY id", [orgId]);
  return r.rows.map((x) => x.action);
}

async function status(orgId: string): Promise<string> {
  return (await getPool().query("SELECT status FROM orgs WHERE id = $1", [orgId])).rows[0].status;
}

describe.skipIf(!hasDb)("admin actions are admin-only", () => {
  it("refuses every admin operation to a normal user", async () => {
    const user = await signUp();
    const victim = await signUp();
    const notAdmin: Actor = { userId: user.userId, email: user.email, userRole: "user" };
    const calls: (() => Promise<unknown>)[] = [
      () => listOrgs(notAdmin),
      () => getOrgDetail(notAdmin, victim.orgId),
      () => recentAudit(notAdmin),
      () => approveOrg(notAdmin, victim.orgId, 5_000_000n),
      () => rejectOrg(notAdmin, victim.orgId, null),
      () => suspendOrg(notAdmin, victim.orgId, "x"),
      () => unsuspendOrg(notAdmin, victim.orgId),
      () => adjustCredits(notAdmin, victim.orgId, 1_000_000n, "x"),
    ];
    for (const c of calls) await expect(c()).rejects.toBeInstanceOf(AuthzError);
    expect(await status(victim.orgId)).toBe("pending_approval");
    expect(await auditActions(victim.orgId)).toEqual([]);
  });

  it("refuses a user who claims the admin role the database does not give them", async () => {
    const user = await signUp();
    const victim = await signUp();
    const forged: Actor = { userId: user.userId, email: user.email, userRole: "admin" };
    await expect(listOrgs(forged)).rejects.toBeInstanceOf(AuthzError);
    await expect(approveOrg(forged, victim.orgId, 5_000_000n)).rejects.toBeInstanceOf(AuthzError);
    await expect(suspendOrg(forged, victim.orgId, "x")).rejects.toBeInstanceOf(AuthzError);
    await expect(adjustCredits(forged, victim.orgId, 1_000_000n, "x")).rejects.toBeInstanceOf(AuthzError);
    expect(await status(victim.orgId)).toBe("pending_approval");
  });
});

describe.skipIf(!hasDb)("signup approval", () => {
  let admin: Actor;
  beforeAll(async () => {
    admin = await makeAdmin();
  });

  it("approve activates the org, posts $5 once, and audits it", async () => {
    const u = await signUp();
    expect((await listOrgs(admin, "pending_approval")).map((o) => o.id)).toContain(u.orgId);
    await approveOrg(admin, u.orgId, 5_000_000n);
    expect(await status(u.orgId)).toBe("active");
    expect((await getBalance(u)).balanceMicros).toBe(5_000_000n);
    await expect(approveOrg(admin, u.orgId, 5_000_000n)).rejects.toThrow(/not waiting for approval/);
    expect((await getBalance(u)).balanceMicros).toBe(5_000_000n);
    expect(await auditActions(u.orgId)).toEqual(["org.approve"]);
  });

  it("reject moves a pending org to rejected with no credit, audited", async () => {
    const u = await signUp();
    await rejectOrg(admin, u.orgId, "spam");
    expect(await status(u.orgId)).toBe("rejected");
    expect((await getBalance(u)).balanceMicros).toBe(0n);
    expect(await auditActions(u.orgId)).toEqual(["org.reject"]);
    await expect(approveOrg(admin, u.orgId, 5_000_000n)).rejects.toBeInstanceOf(AuthzError);
  });
});

describe.skipIf(!hasDb)("suspension", () => {
  it("suspends at once, notifies the gateway, and can be undone; both audited", async () => {
    const admin = await makeAdmin();
    const u = await activeUser(admin);
    const listener = new Client({ connectionString: process.env.DATABASE_URL });
    await listener.connect();
    try {
      await listener.query("LISTEN loom_auth_changed");
      const got = new Promise<Record<string, string>>((resolve) => {
        listener.on("notification", (n) => {
          const p = JSON.parse(n.payload ?? "{}");
          if (p.kind === "org" && p.org_id === u.orgId) resolve(p);
        });
      });
      await suspendOrg(admin, u.orgId, "abuse report");
      expect(await got).toMatchObject({ kind: "org", org_id: u.orgId, status: "suspended" });
    } finally {
      await listener.end();
    }
    expect((await loadOrgContext(u.userId))?.orgStatus).toBe("suspended");
    await expect(suspendOrg(admin, u.orgId, "again")).rejects.toBeInstanceOf(AuthzError);
    await unsuspendOrg(admin, u.orgId);
    expect(await status(u.orgId)).toBe("active");
    expect(await auditActions(u.orgId)).toEqual(["org.approve", "org.suspend", "org.unsuspend"]);
    const detail = await getOrgDetail(admin, u.orgId);
    expect(detail?.audit[1].details).toMatchObject({ reason: "abuse report" });
  });
});

describe.skipIf(!hasDb)("credit adjustments", () => {
  it("posts grants and corrections through the ledger, keyed by the audit row", async () => {
    const admin = await makeAdmin();
    const u = await activeUser(admin);
    await adjustCredits(admin, u.orgId, 2_500_000n, "goodwill");
    await adjustCredits(admin, u.orgId, -1_000_000n, "double credit fix");
    expect((await getBalance(u)).balanceMicros).toBe(6_500_000n);

    const tx = await getPool().query(
      `SELECT t.idempotency_key, t.memo, array_agg(e.account_id::text ORDER BY e.amount_micros) AS accounts
       FROM ledger_transactions t JOIN ledger_entries e ON e.transaction_id = t.id
       WHERE t.org_id = $1 AND t.kind = 'adjustment' GROUP BY t.id ORDER BY t.created_at`,
      [u.orgId],
    );
    const audits = await getPool().query(
      "SELECT id, details FROM admin_audit_log WHERE target_id = $1 AND action = 'org.credit_adjust' ORDER BY id",
      [u.orgId],
    );
    expect(tx.rows.map((r) => r.idempotency_key)).toEqual(audits.rows.map((r) => `admin:${r.id}`));
    expect(tx.rows[0].accounts).toContain("00000000-0000-0000-0000-000000000003"); // grant vs promo_expense
    expect(tx.rows[1].accounts).toContain("00000000-0000-0000-0000-000000000001"); // correction vs revenue
    expect(audits.rows[1].details).toMatchObject({ amount_micros: "-1000000", kind: "correction" });
  });

  it("refuses zero, oversized and unexplained adjustments", async () => {
    const admin = await makeAdmin();
    const u = await activeUser(admin);
    await expect(adjustCredits(admin, u.orgId, 0n, "x")).rejects.toBeInstanceOf(RangeError);
    await expect(adjustCredits(admin, u.orgId, 1_000_000n, "  ")).rejects.toBeInstanceOf(RangeError);
    await expect(adjustCredits(admin, u.orgId, 20_000_000_000n, "x")).rejects.toBeInstanceOf(RangeError);
    expect((await getBalance(u)).balanceMicros).toBe(5_000_000n);
  });
});
