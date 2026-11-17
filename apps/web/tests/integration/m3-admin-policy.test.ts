// W5 (admin policy), docs/m3-review-fixes-spec.md: LOOM_ADMIN_EMAILS grants admin only
// with LOOM_ENV=dev, and outside dev an admin cannot approve, or adjust the credit of, an
// org they are a member of. The sign-in grant is tested through Auth.js's `events.signIn`
// (src/auth.config.ts), approval and credit through the admin actions (and approveOrg in
// the data layer), the startup warning through `register()` (src/instrumentation.ts);
// only the identity layer (`auth()` from @/auth) is faked.

import { randomUUID } from "node:crypto";

import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import { authConfig } from "@/auth.config";
import { register } from "@/instrumentation";
import type { Actor, OrgContext } from "@/lib/authz";
import { getPool } from "@/lib/db";
import { approveOrg } from "@/lib/data/admin";
import { loadActor, loadOrgContext } from "@/lib/data/orgs";

import { activeUser, adapter, addMember, email, hasDb, makeAdmin, signUp } from "./helpers";
import { adminSnapshot, creditMicros, form } from "./m3-helpers";

const session = vi.hoisted(() => ({ userId: null as string | null }));

vi.mock("@/auth", () => ({
  auth: vi.fn(async () => (session.userId ? { user: { id: session.userId } } : null)),
  signIn: vi.fn(async () => undefined),
  signOut: vi.fn(async () => undefined),
}));
vi.mock("next-auth", () => ({ AuthError: class AuthError extends Error {} }));
vi.mock("next/cache", () => ({ revalidatePath: vi.fn() }));

const { adjustCreditsAction, orgStatusAction } = await import("@/app/admin/actions");

const savedEnv = { LOOM_ENV: process.env.LOOM_ENV, LOOM_ADMIN_EMAILS: process.env.LOOM_ADMIN_EMAILS };

function setLoomEnv(v: string | undefined) {
  if (v === undefined) delete process.env.LOOM_ENV;
  else process.env.LOOM_ENV = v;
}

afterEach(() => {
  setLoomEnv(savedEnv.LOOM_ENV);
  process.env.LOOM_ADMIN_EMAILS = savedEnv.LOOM_ADMIN_EMAILS;
});

/** Auth.js's sign-in event, as it fires after a magic-link sign-in. */
async function signInEvent(userId: string, addr: string): Promise<void> {
  await authConfig().events!.signIn!({
    user: { id: userId, email: addr },
    account: { provider: "nodemailer", type: "email", providerAccountId: addr },
    isNewUser: false,
  } as never);
}

async function role(userId: string): Promise<string> {
  return (await getPool().query("SELECT role FROM users WHERE id = $1", [userId])).rows[0].role;
}

async function grantAudits(userId: string): Promise<number> {
  const r = await getPool().query(
    "SELECT count(*)::int AS n FROM admin_audit_log WHERE action = 'user.grant_admin' AND target_id = $1",
    [userId],
  );
  return r.rows[0].n;
}

async function orgStatus(orgId: string): Promise<string> {
  return (await getPool().query("SELECT status FROM orgs WHERE id = $1", [orgId])).rows[0].status;
}

/** A signed-up user (owner of their pending org) whom the database makes an admin. */
async function adminWithOwnOrg(): Promise<{ actor: Actor; own: OrgContext }> {
  const own = await signUp(email("selfadmin"));
  await getPool().query("UPDATE users SET role = 'admin' WHERE id = $1", [own.userId]);
  return { actor: (await loadActor(own.userId))!, own };
}

function adjustForm(orgId: string, amount = "25"): FormData {
  return form({ org_id: orgId, amount, memo: "self service", form_token: randomUUID() });
}

describe.skipIf(!hasDb)("W5: LOOM_ADMIN_EMAILS grants admin at sign-in only in dev", () => {
  it.each([
    ["unset", undefined],
    ["prod", "prod"],
  ])("W5: with LOOM_ENV %s, a listed email signing in stays a user, with no audit row", async (_label, env) => {
    setLoomEnv(env);
    const addr = email("listed");
    process.env.LOOM_ADMIN_EMAILS = `someone@example.com, ${addr.toUpperCase()}`;
    const user = await adapter().createUser!({ id: "", email: addr, emailVerified: new Date(), name: null, image: null });

    await signInEvent(user.id, addr);

    expect(await role(user.id)).toBe("user");
    expect(await grantAudits(user.id)).toBe(0);
    expect((await loadOrgContext(user.id))?.userRole).toBe("user");
  });

  it("W5: with LOOM_ENV=dev, a listed email signing in becomes an admin, audited", async () => {
    setLoomEnv("dev");
    const addr = email("listed-dev");
    process.env.LOOM_ADMIN_EMAILS = addr;
    const user = await adapter().createUser!({ id: "", email: addr, emailVerified: new Date(), name: null, image: null });

    await signInEvent(user.id, addr);

    expect(await role(user.id)).toBe("admin");
    expect(await grantAudits(user.id)).toBe(1);
  });

  it("W5: outside dev, an admin already granted keeps the role at sign-in", async () => {
    setLoomEnv("prod");
    const addr = email("granted");
    process.env.LOOM_ADMIN_EMAILS = addr;
    const r = await getPool().query<{ id: string }>("INSERT INTO users (email, role) VALUES ($1, 'admin') RETURNING id", [
      addr,
    ]);
    await signInEvent(r.rows[0].id, addr);
    expect(await role(r.rows[0].id)).toBe("admin");
  });
});

// "Outside dev" is any LOOM_ENV other than dev, unset included.
describe.skipIf(!hasDb).each([
  ["unset", undefined],
  ["prod", "prod"],
])("W5: with LOOM_ENV %s (outside dev), no admin acts on an org they belong to", (_label, env) => {
  let otherAdmin: Actor;

  beforeAll(async () => {
    otherAdmin = await makeAdmin();
  });

  beforeEach(() => {
    setLoomEnv(env);
  });

  it("W5: refuses an admin approving their own org (action and data layer), changing nothing", async () => {
    const { actor, own } = await adminWithOwnOrg();
    session.userId = actor.userId;

    const before = await adminSnapshot();
    const r = await orgStatusAction(undefined, form({ org_id: own.orgId, op: "approve" }));
    expect(r?.error).toBeTruthy();
    expect(r?.ok).toBeUndefined();
    await expect(approveOrg(actor, own.orgId, 5_000_000n)).rejects.toThrow();
    expect(await adminSnapshot()).toEqual(before);
    expect(await orgStatus(own.orgId)).toBe("pending_approval");
    expect(await creditMicros(own.orgId)).toBe(0n);
  });

  it("W5: refuses an admin approving an org they are a plain member of", async () => {
    const pending = await signUp();
    const member = await addMember(pending.orgId, "member");
    await getPool().query("UPDATE users SET role = 'admin' WHERE id = $1", [member.userId]);
    const actor = (await loadActor(member.userId))!;
    session.userId = actor.userId;

    const before = await adminSnapshot();
    const r = await orgStatusAction(undefined, form({ org_id: pending.orgId, op: "approve" }));
    expect(r?.error).toBeTruthy();
    await expect(approveOrg(actor, pending.orgId, 5_000_000n)).rejects.toThrow();
    expect(await adminSnapshot()).toEqual(before);
  });

  it("W5: refuses an admin adjusting their own org's credit, with no audit row or ledger entry", async () => {
    const { actor, own } = await adminWithOwnOrg();
    // Their org is active (another admin approved it), so only the membership is in the way.
    await approveOrg(otherAdmin, own.orgId, 5_000_000n);
    session.userId = actor.userId;

    const before = await adminSnapshot();
    const credit = await creditMicros(own.orgId);
    for (const amount of ["25", "-1"]) {
      const r = await adjustCreditsAction(undefined, adjustForm(own.orgId, amount));
      expect(r?.error, amount).toBeTruthy();
      expect(r?.ok).toBeUndefined();
    }
    expect(await adminSnapshot()).toEqual(before);
    expect(await creditMicros(own.orgId)).toBe(credit);
  });

  it("W5: the same admin can still approve and credit another org", async () => {
    const { actor } = await adminWithOwnOrg();
    session.userId = actor.userId;

    const pending = await signUp();
    const approved = await orgStatusAction(undefined, form({ org_id: pending.orgId, op: "approve" }));
    expect(approved?.error).toBeUndefined();
    expect(approved?.ok).toBeTruthy();
    expect(await orgStatus(pending.orgId)).toBe("active");

    const active = await activeUser(otherAdmin);
    const credit = await creditMicros(active.orgId);
    const adjusted = await adjustCreditsAction(undefined, adjustForm(active.orgId, "3"));
    expect(adjusted?.error).toBeUndefined();
    expect(adjusted?.ok).toBeTruthy();
    expect(await creditMicros(active.orgId)).toBe(credit + 3_000_000n);
  });
});

describe.skipIf(!hasDb)("W5: in dev, an admin may still approve and credit their own org", () => {
  beforeEach(() => {
    setLoomEnv("dev");
  });

  it("W5: with LOOM_ENV=dev, the bootstrap admin approves their own org", async () => {
    const { actor, own } = await adminWithOwnOrg();
    session.userId = actor.userId;
    const r = await orgStatusAction(undefined, form({ org_id: own.orgId, op: "approve" }));
    expect(r?.error).toBeUndefined();
    expect(r?.ok).toBeTruthy();
    expect(await orgStatus(own.orgId)).toBe("active");
  });

  it("W5: with LOOM_ENV=dev, an admin adjusts their own org's credit", async () => {
    const { actor, own } = await adminWithOwnOrg();
    await approveOrg(actor, own.orgId, 5_000_000n);
    session.userId = actor.userId;
    const credit = await creditMicros(own.orgId);
    const r = await adjustCreditsAction(undefined, adjustForm(own.orgId, "4"));
    expect(r?.error).toBeUndefined();
    expect(r?.ok).toBeTruthy();
    expect(await creditMicros(own.orgId)).toBe(credit + 4_000_000n);
  });
});

// Startup (src/instrumentation.ts): outside dev, a set LOOM_ADMIN_EMAILS is ignored, and
// the server says so. The environment passes every other production check, so the warning
// is all there is to see (the server still starts).
describe("W5: the startup warning for LOOM_ADMIN_EMAILS outside dev", () => {
  const PROD_OK = {
    NEXT_RUNTIME: "nodejs",
    AUTH_SECRET: "a-real-random-secret",
    AUTH_URL: "https://app.loom.example",
    LOOM_BILLING_INTERNAL_TOKEN: "a-real-token",
  };

  /** What register() logs as a warning or an error, with LOOM_ENV and LOOM_ADMIN_EMAILS as given. */
  async function startupLog(loomEnvValue: string | undefined, adminEmails: string | undefined): Promise<string[]> {
    const exit = vi.spyOn(process, "exit").mockImplementation(() => undefined as never);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      for (const [k, v] of Object.entries(PROD_OK)) vi.stubEnv(k, v);
      vi.stubEnv("LOOM_ENV", loomEnvValue);
      vi.stubEnv("LOOM_ADMIN_EMAILS", adminEmails);
      await register();
      expect(exit).not.toHaveBeenCalled();
      return [...warn.mock.calls, ...error.mock.calls].map((args) => args.map(String).join(" "));
    } finally {
      vi.unstubAllEnvs();
      exit.mockRestore();
      warn.mockRestore();
      error.mockRestore();
    }
  }

  it.each([
    ["unset", undefined],
    ["prod", "prod"],
  ])("W5: with LOOM_ENV %s and LOOM_ADMIN_EMAILS set, register() logs a warning naming it", async (_label, env) => {
    const logged = await startupLog(env, "boss@example.com");
    expect(logged).toContainEqual(expect.stringMatching(/LOOM_ADMIN_EMAILS/));
  });

  it("W5: with LOOM_ENV prod and LOOM_ADMIN_EMAILS unset, register() logs no such warning", async () => {
    const logged = await startupLog("prod", undefined);
    expect(logged.filter((m) => /LOOM_ADMIN_EMAILS/.test(m))).toEqual([]);
  });

  it("W5: with LOOM_ENV=dev, register() logs no warning about LOOM_ADMIN_EMAILS", async () => {
    const logged = await startupLog("dev", "boss@example.com");
    expect(logged.filter((m) => /LOOM_ADMIN_EMAILS/.test(m))).toEqual([]);
  });
});
