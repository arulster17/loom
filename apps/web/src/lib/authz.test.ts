import { describe, expect, it } from "vitest";

import {
  assertOrgActive,
  assertPlatformAdmin,
  AuthzError,
  canCreateKeys,
  canManageBilling,
  canRevokeKey,
  isBootstrapAdmin,
  isUuid,
  parseAdminEmails,
  type OrgContext,
} from "./authz";

function ctx(over: Partial<OrgContext> = {}): OrgContext {
  return {
    userId: "11111111-1111-1111-1111-111111111111",
    email: "a@example.com",
    userRole: "user",
    orgId: "22222222-2222-2222-2222-222222222222",
    orgName: "a's org",
    orgRole: "owner",
    orgStatus: "active",
    ...over,
  };
}

describe("platform admin", () => {
  it("only users.role = admin passes", () => {
    expect(() => assertPlatformAdmin({ userRole: "admin" })).not.toThrow();
    expect(() => assertPlatformAdmin({ userRole: "user" })).toThrow(AuthzError);
  });
});

describe("org status", () => {
  it("keys and playground need an active org", () => {
    expect(canCreateKeys(ctx())).toBe(true);
    for (const s of ["pending_approval", "suspended", "rejected"] as const) {
      expect(canCreateKeys(ctx({ orgStatus: s }))).toBe(false);
      expect(() => assertOrgActive(ctx({ orgStatus: s }))).toThrow(AuthzError);
    }
  });

  it("explains a pending account in plain words", () => {
    expect(() => assertOrgActive(ctx({ orgStatus: "pending_approval" }))).toThrow(/waiting for approval/);
  });
});

describe("key revocation", () => {
  const me = ctx().userId;
  const someoneElse = "33333333-3333-3333-3333-333333333333";

  it("owners and org admins revoke any key in their org", () => {
    expect(canRevokeKey(ctx({ orgRole: "owner" }), { createdBy: someoneElse })).toBe(true);
    expect(canRevokeKey(ctx({ orgRole: "admin" }), { createdBy: null })).toBe(true);
  });

  it("members revoke only their own keys", () => {
    expect(canRevokeKey(ctx({ orgRole: "member" }), { createdBy: me })).toBe(true);
    expect(canRevokeKey(ctx({ orgRole: "member" }), { createdBy: someoneElse })).toBe(false);
    expect(canRevokeKey(ctx({ orgRole: "member" }), { createdBy: null })).toBe(false);
  });
});

describe("billing", () => {
  it("owners and org admins of an active org manage billing; members do not", () => {
    expect(canManageBilling(ctx({ orgRole: "owner" }))).toBe(true);
    expect(canManageBilling(ctx({ orgRole: "admin" }))).toBe(true);
    expect(canManageBilling(ctx({ orgRole: "member" }))).toBe(false);
    expect(canManageBilling(ctx({ orgStatus: "suspended" }))).toBe(false);
  });
});

describe("admin bootstrap emails", () => {
  it("parses comma or space separated, case-insensitively", () => {
    expect([...parseAdminEmails(" A@x.com, b@y.com  c@z.com,")]).toEqual(["a@x.com", "b@y.com", "c@z.com"]);
    expect(isBootstrapAdmin("B@Y.com", "a@x.com,b@y.com")).toBe(true);
    expect(isBootstrapAdmin("b@y.com", undefined)).toBe(false);
    expect(isBootstrapAdmin(null, "b@y.com")).toBe(false);
    expect(isBootstrapAdmin("", "")).toBe(false);
  });
});

describe("ids from requests", () => {
  it("accepts uuids only", () => {
    expect(isUuid("6f1c2a4e-1b2c-4d3e-8f90-123456789abc")).toBe(true);
    expect(isUuid("6f1c2a4e")).toBe(false);
    expect(isUuid("' OR 1=1 --")).toBe(false);
    expect(isUuid(42)).toBe(false);
  });
});
