// Authorization of every exported server action, called the way Next.js calls them:
// only the identity layer is faked (`auth()` from @/auth returns a session for the user
// the test picks); the session's org, role and status are then read from the real
// database by src/lib/session.ts, and the actions run their real data functions. Every
// refusal is checked to leave the database exactly as it was.

import { randomUUID } from "node:crypto";

import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { Actor, OrgContext } from "@/lib/authz";
import { getPool } from "@/lib/db";
import { createKey } from "@/lib/data/keys";
import { getPlaygroundKey } from "@/lib/data/playground";

import { activeUser, addMember, hasDb, makeAdmin, signUp } from "./helpers";

const session = vi.hoisted(() => ({ userId: null as string | null }));

vi.mock("@/auth", () => ({
  auth: vi.fn(async () => (session.userId ? { user: { id: session.userId } } : null)),
  signIn: vi.fn(async () => undefined),
  signOut: vi.fn(async () => undefined),
}));
// The real package expects the Next.js bundler (it imports `next/server` unresolved).
vi.mock("next-auth", () => ({ AuthError: class AuthError extends Error {} }));
// revalidatePath needs a Next.js request store; nothing to revalidate in a test.
vi.mock("next/cache", () => ({ revalidatePath: vi.fn() }));
// Billing is another service: record what the action would ask it for.
vi.mock("@/lib/services", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/services")>()),
  createCheckout: vi.fn(async () => "https://checkout.stripe.com/c/pay/cs_test"),
}));

const { signOut } = await import("@/auth");
const { createCheckout } = await import("@/lib/services");
const { signOutAction } = await import("@/app/actions");
const { adjustCreditsAction, orgStatusAction } = await import("@/app/admin/actions");
const { autoRechargeAction, checkoutAction } = await import("@/app/dashboard/billing/actions");
const { createKeyAction, revokeKeyAction } = await import("@/app/dashboard/keys/actions");

function as(user: { userId: string } | null) {
  session.userId = user?.userId ?? null;
}

function form(fields: Record<string, string>): FormData {
  const f = new FormData();
  for (const [k, v] of Object.entries(fields)) f.set(k, v);
  return f;
}

/** Everything a web action can write. Compared before and after each refusal. */
async function snapshot(): Promise<Record<string, unknown>> {
  const r = await getPool().query(
    `SELECT (SELECT count(*) FROM users)::int AS users,
            (SELECT count(*) FROM orgs)::int AS orgs,
            (SELECT string_agg(id || ':' || status || ':' || coalesce(suspended_reason, ''), ',' ORDER BY id) FROM orgs) AS org_status,
            (SELECT string_agg(id || ':' || role, ',' ORDER BY id) FROM users) AS roles,
            (SELECT count(*) FROM api_keys)::int AS keys,
            (SELECT count(*) FROM api_keys WHERE revoked_at IS NOT NULL)::int AS revoked,
            (SELECT count(*) FROM ledger_transactions)::int AS ledger,
            (SELECT count(*) FROM admin_audit_log)::int AS audit,
            (SELECT string_agg(org_id || ':' || enabled || ':' || threshold_micros || ':' || amount_micros, ',' ORDER BY org_id)
             FROM auto_recharge) AS auto_recharge`,
  );
  return r.rows[0];
}

/** Runs the action and asserts it changed nothing in the database. */
async function refusedWithoutWrites<T>(call: () => Promise<T>): Promise<T> {
  const before = await snapshot();
  const result = await call();
  expect(await snapshot()).toEqual(before);
  return result;
}

async function orgStatus(orgId: string): Promise<string> {
  return (await getPool().query("SELECT status FROM orgs WHERE id = $1", [orgId])).rows[0].status;
}

describe.skipIf(!hasDb)("server actions: authorization", () => {
  let admin: Actor;
  let owner: OrgContext; // owner of an active org
  let other: OrgContext; // owner of another active org
  let member: OrgContext; // plain member of `owner`'s org
  let orgAdmin: OrgContext; // org-level admin of `owner`'s org (not a platform admin)
  let pending: OrgContext;
  let suspended: OrgContext;
  let rejected: OrgContext;

  beforeAll(async () => {
    process.env.AUTH_URL = "http://localhost:3000"; // checkout return URLs without headers()
    admin = await makeAdmin();
    owner = await activeUser(admin);
    other = await activeUser(admin);
    member = await addMember(owner.orgId, "member");
    orgAdmin = await addMember(owner.orgId, "admin");
    pending = await signUp();
    suspended = await activeUser(admin);
    await getPool().query("UPDATE orgs SET status = 'suspended', suspended_reason = 'test' WHERE id = $1", [suspended.orgId]);
    rejected = await signUp();
    await getPool().query("UPDATE orgs SET status = 'rejected' WHERE id = $1", [rejected.orgId]);
  });

  beforeEach(() => {
    vi.mocked(createCheckout).mockClear();
    vi.mocked(signOut).mockClear();
  });

  afterEach(() => as(null));

  describe("admin actions", () => {
    async function adminCalls() {
      const victimPending = await signUp();
      const victimActive = await activeUser(admin);
      const victimSuspended = await activeUser(admin);
      await getPool().query("UPDATE orgs SET status = 'suspended' WHERE id = $1", [victimSuspended.orgId]);
      return [
        () => orgStatusAction(undefined, form({ org_id: victimPending.orgId, op: "approve" })),
        () => orgStatusAction(undefined, form({ org_id: victimPending.orgId, op: "reject", reason: "no" })),
        () => orgStatusAction(undefined, form({ org_id: victimActive.orgId, op: "suspend", reason: "abuse" })),
        () => orgStatusAction(undefined, form({ org_id: victimSuspended.orgId, op: "unsuspend" })),
        () => adjustCreditsAction(undefined, form({ org_id: victimActive.orgId, amount: "100", memo: "gift" })),
        () => adjustCreditsAction(undefined, form({ org_id: victimActive.orgId, amount: "-5", memo: "take" })),
      ];
    }

    it.each([
      ["signed out", () => null],
      ["an org owner", () => owner],
      ["an org-level admin (not a platform admin)", () => orgAdmin],
      ["a plain member", () => member],
      ["a pending signup", () => pending],
    ])("refuse %s, writing nothing", async (_label, caller) => {
      as(caller());
      for (const call of await adminCalls()) {
        expect(await refusedWithoutWrites(call)).toEqual({ error: "Admins only." });
      }
    });

    it("refuse an admin whose role was revoked while their session is still valid", async () => {
      const former = await makeAdmin();
      await getPool().query("UPDATE users SET role = 'user' WHERE id = $1", [former.userId]);
      as(former);
      for (const call of await adminCalls()) {
        expect(await refusedWithoutWrites(call)).toEqual({ error: "Admins only." });
      }
    });

    it("let a platform admin act (the refusals above are not vacuous)", async () => {
      const victim = await signUp();
      as(admin);
      const r = await orgStatusAction(undefined, form({ org_id: victim.orgId, op: "approve" }));
      expect(r?.ok).toMatch(/^Approved/);
      expect(await orgStatus(victim.orgId)).toBe("active");
    });
  });

  describe("API key actions", () => {
    it("createKeyAction refuses a signed-out caller", async () => {
      as(null);
      expect(await refusedWithoutWrites(() => createKeyAction(undefined, form({ name: "x" })))).toEqual({
        ok: false,
        error: "Sign in again.",
      });
    });

    it.each([
      ["pending", () => pending],
      ["suspended", () => suspended],
      ["rejected", () => rejected],
    ])("createKeyAction refuses a %s org", async (_label, caller) => {
      as(caller());
      const r = await refusedWithoutWrites(() => createKeyAction(undefined, form({ name: "x", spend_limit: "5" })));
      expect(r).toMatchObject({ ok: false });
    });

    it("createKeyAction creates the key in the caller's own org, whatever org the form names", async () => {
      as(owner);
      const r = await createKeyAction(undefined, form({ name: "mine", org_id: other.orgId }));
      expect(r).toMatchObject({ ok: true, name: "mine" });
      const row = await getPool().query("SELECT org_id, created_by FROM api_keys WHERE prefix = $1", [
        r && r.ok ? r.prefix : "",
      ]);
      expect(row.rows[0]).toEqual({ org_id: owner.orgId, created_by: owner.userId });
    });

    it("revokeKeyAction refuses a signed-out caller", async () => {
      const key = await createKey(owner, { name: "k", spendLimitMicros: null });
      as(null);
      expect(await refusedWithoutWrites(() => revokeKeyAction(undefined, form({ key_id: key.id })))).toEqual({
        error: "Sign in again.",
      });
    });

    it("revokeKeyAction cannot reach another org's key: it is not found", async () => {
      const victim = await createKey(other, { name: "victim", spendLimitMicros: null });
      for (const caller of [owner, orgAdmin, member, pending, suspended]) {
        as(caller);
        expect(await refusedWithoutWrites(() => revokeKeyAction(undefined, form({ key_id: victim.id })))).toEqual({
          error: "Key not found.",
        });
      }
    });

    it("revokeKeyAction refuses a plain member revoking someone else's key", async () => {
      const key = await createKey(owner, { name: "owner's", spendLimitMicros: null });
      as(member);
      const r = await refusedWithoutWrites(() => revokeKeyAction(undefined, form({ key_id: key.id })));
      expect(r?.error).toMatch(/creator or an org owner or admin/);
    });

    it("revokeKeyAction treats junk and unknown ids as not found", async () => {
      as(owner);
      for (const id of ["", "not-a-uuid", "'; DROP TABLE api_keys; --", randomUUID()]) {
        expect(await refusedWithoutWrites(() => revokeKeyAction(undefined, form({ key_id: id })))).toEqual({
          error: "Key not found.",
        });
      }
    });

    it("revokeKeyAction lets the owner revoke their own org's key", async () => {
      const key = await createKey(owner, { name: "to-revoke", spendLimitMicros: null });
      as(owner);
      expect(await revokeKeyAction(undefined, form({ key_id: key.id }))).toBeUndefined();
      const row = await getPool().query("SELECT revoked_at FROM api_keys WHERE id = $1", [key.id]);
      expect(row.rows[0].revoked_at).not.toBeNull();
    });
  });

  describe("billing actions", () => {
    const topup = { amount: "20" };
    const recharge = { enabled: "on", threshold: "5", amount: "20" };

    it.each([
      ["signed out", () => null, "Sign in again."],
      ["a plain member", () => member, /owners and admins/],
      ["a pending org's owner", () => pending, /owners and admins/],
      ["a suspended org's owner", () => suspended, /owners and admins/],
      ["a rejected org's owner", () => rejected, /owners and admins/],
    ])("checkoutAction refuses %s without calling billing", async (_label, caller, message) => {
      as(caller());
      const r = await refusedWithoutWrites(() => checkoutAction(undefined, form(topup)));
      expect(r?.error).toMatch(message);
      expect(createCheckout).not.toHaveBeenCalled();
    });

    it("checkoutAction always charges the caller's own org, whatever org the form names", async () => {
      as(owner);
      await expect(checkoutAction(undefined, form({ ...topup, org_id: other.orgId }))).rejects.toThrow("NEXT_REDIRECT");
      expect(createCheckout).toHaveBeenCalledTimes(1);
      expect(vi.mocked(createCheckout).mock.calls[0][0]).toMatchObject({
        orgId: owner.orgId,
        customerEmail: owner.email,
        amountMicros: 20_000_000n,
      });
    });

    it.each([
      ["signed out", () => null, "Sign in again."],
      ["a plain member", () => member, /owners and admins/],
      ["a pending org's owner", () => pending, /owners and admins/],
      ["a suspended org's owner", () => suspended, /owners and admins/],
      ["a rejected org's owner", () => rejected, /owners and admins/],
    ])("autoRechargeAction refuses %s, writing nothing", async (_label, caller, message) => {
      as(caller());
      const r = await refusedWithoutWrites(() => autoRechargeAction(undefined, form(recharge)));
      expect(r?.error).toMatch(message);
    });

    it("autoRechargeAction saves for the caller's own org only, whatever org the form names", async () => {
      as(orgAdmin);
      expect(await autoRechargeAction(undefined, form({ ...recharge, org_id: other.orgId }))).toEqual({ ok: "Saved." });
      const rows = await getPool().query("SELECT org_id FROM auto_recharge WHERE org_id = ANY($1)", [
        [owner.orgId, other.orgId],
      ]);
      expect(rows.rows).toEqual([{ org_id: owner.orgId }]);
    });
  });

  describe("sign-out", () => {
    it("signed out: revokes nothing and still clears the session", async () => {
      as(null);
      await refusedWithoutWrites(() => expect(signOutAction()).rejects.toThrow("NEXT_REDIRECT"));
      expect(signOut).toHaveBeenCalledTimes(1);
    });

    it("revokes only the caller's playground keys, never another user's", async () => {
      const a = await activeUser(admin);
      const b = await activeUser(admin);
      const teammate = await addMember(a.orgId, "member");
      await getPlaygroundKey(a);
      await getPlaygroundKey(b);
      await getPlaygroundKey(teammate);
      as(a);
      await expect(signOutAction()).rejects.toThrow("NEXT_REDIRECT");
      const r = await getPool().query<{ created_by: string; revoked: boolean }>(
        `SELECT created_by, revoked_at IS NOT NULL AS revoked FROM api_keys
         WHERE 'playground' = ANY (scopes) AND created_by = ANY($1)`,
        [[a.userId, b.userId, teammate.userId]],
      );
      const byUser = Object.fromEntries(r.rows.map((x) => [x.created_by, x.revoked]));
      expect(byUser).toEqual({ [a.userId]: true, [b.userId]: false, [teammate.userId]: false });
    });
  });
});
