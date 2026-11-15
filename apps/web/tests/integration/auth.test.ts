// The Auth.js Postgres adapter against Loom's schema (uuid ids, citext email, role),
// and the signup flow built on it: org creation and the admin bootstrap.

import { randomUUID } from "node:crypto";

import { describe, expect, it } from "vitest";

import { getPool } from "@/lib/db";
import { ensureUserOrg, loadOrgContext, syncBootstrapAdmin } from "@/lib/data/orgs";

import { adapter, email, hasDb, signUp } from "./helpers";

const UUID = /^[0-9a-f-]{36}$/;

describe.skipIf(!hasDb)("@auth/pg-adapter on the uuid schema", () => {
  it("creates users with database-generated uuid ids and reads them back", async () => {
    const a = adapter();
    const addr = email();
    const passedId = randomUUID(); // Auth.js core passes one; the adapter must ignore it
    const user = await a.createUser!({ id: passedId, email: addr, emailVerified: new Date(), name: null, image: null });
    expect(user.id).toMatch(UUID);
    expect(user.id).not.toBe(passedId);
    expect((await a.getUser!(user.id))?.email).toBe(addr);
    expect(await a.getUser!("not-a-uuid")).toBeNull();
    // citext: the lookup Auth.js does at sign-in is case-insensitive.
    expect((await a.getUserByEmail!(addr.toUpperCase()))?.id).toBe(user.id);
    const updated = await a.updateUser!({ id: user.id, name: "Ada" });
    expect(updated.name).toBe("Ada");
  });

  it("links OAuth accounts and finds users by account", async () => {
    const a = adapter();
    const user = await a.createUser!({ id: "", email: email(), emailVerified: null, name: null, image: null });
    const providerAccountId = `g-${randomUUID()}`;
    const linked = await a.linkAccount!({
      userId: user.id,
      type: "oidc",
      provider: "google",
      providerAccountId,
      access_token: "at",
      expires_at: 1_900_000_000,
      token_type: "bearer",
      scope: "openid email",
      id_token: "it",
    });
    expect(linked?.expires_at).toBe(1_900_000_000);
    expect((await a.getUserByAccount!({ provider: "google", providerAccountId }))?.id).toBe(user.id);
  });

  it("stores database sessions keyed by uuid user ids", async () => {
    const a = adapter();
    const user = await a.createUser!({ id: "", email: email(), emailVerified: null, name: null, image: null });
    const sessionToken = randomUUID();
    const expires = new Date(Date.now() + 3600_000);
    const s = await a.createSession!({ sessionToken, userId: user.id, expires });
    expect(s.userId).toBe(user.id);
    const both = await a.getSessionAndUser!(sessionToken);
    expect(both?.user.id).toBe(user.id);
    expect((both?.user as unknown as { role: string }).role).toBe("user");
    await a.updateSession!({ sessionToken, expires: new Date(Date.now() + 7200_000) });
    await a.deleteSession!(sessionToken);
    expect(await a.getSessionAndUser!(sessionToken)).toBeNull();
  });

  it("issues and consumes magic-link verification tokens once", async () => {
    const a = adapter();
    const identifier = email();
    const token = randomUUID();
    await a.createVerificationToken!({ identifier, token, expires: new Date(Date.now() + 600_000) });
    expect((await a.useVerificationToken!({ identifier, token }))?.identifier).toBe(identifier);
    expect(await a.useVerificationToken!({ identifier, token })).toBeNull();
  });
});

describe.skipIf(!hasDb)("signup", () => {
  it("gives a new user their own org, pending approval, as owner", async () => {
    const ctx = await signUp(email("alice"));
    expect(ctx.orgStatus).toBe("pending_approval");
    expect(ctx.orgRole).toBe("owner");
    expect(ctx.orgName).toMatch(/^alice-.*'s org$/);
    const credit = await getPool().query("SELECT loom_org_credit_total($1) AS c", [ctx.orgId]);
    expect(credit.rows[0].c).toBe("0"); // the $5 waits for approval
  });

  it("creates exactly one org even when sign-in events race", async () => {
    const user = await adapter().createUser!({ id: "", email: email(), emailVerified: null, name: null, image: null });
    const ids = await Promise.all(Array.from({ length: 5 }, () => ensureUserOrg(user.id)));
    expect(new Set(ids).size).toBe(1);
    const n = await getPool().query("SELECT count(*)::int AS n FROM org_members WHERE user_id = $1", [user.id]);
    expect(n.rows[0].n).toBe(1);
  });

  it("grants admin to LOOM_ADMIN_EMAILS at sign-in, audited, once", async () => {
    const user = await adapter().createUser!({ id: "", email: "Boss@Example.com", emailVerified: null, name: null, image: null });
    expect(await syncBootstrapAdmin(user.id, "boss@example.com")).toBe(true);
    expect(await syncBootstrapAdmin(user.id, "boss@example.com")).toBe(false);
    const role = await getPool().query("SELECT role FROM users WHERE id = $1", [user.id]);
    expect(role.rows[0].role).toBe("admin");
    const log = await getPool().query(
      "SELECT action FROM admin_audit_log WHERE target_type = 'user' AND target_id = $1",
      [user.id],
    );
    expect(log.rows.map((r) => r.action)).toEqual(["user.grant_admin"]);
  });

  it("does not grant admin to anyone else", async () => {
    const ctx = await signUp();
    expect(await syncBootstrapAdmin(ctx.userId, "boss@example.com")).toBe(false);
    expect((await loadOrgContext(ctx.userId))?.userRole).toBe("user");
  });
});
