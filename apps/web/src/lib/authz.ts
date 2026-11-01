// Authorization rules, as pure functions over a caller context. The data layer
// (src/lib/data/*) calls these before every read or write, and every query is also
// scoped by the caller's org id, so a check that is forgotten still cannot reach
// another org's rows.

export type UserRole = "user" | "admin";
export type OrgRole = "owner" | "admin" | "member";
export type OrgStatus = "pending_approval" | "active" | "suspended" | "rejected";

/** A signed-in user. `userRole = 'admin'` is a Loom platform admin. */
export interface Actor {
  userId: string;
  email: string;
  userRole: UserRole;
}

/** A signed-in user acting inside one org they belong to. */
export interface OrgContext extends Actor {
  orgId: string;
  orgName: string;
  orgRole: OrgRole;
  orgStatus: OrgStatus;
}

export class AuthzError extends Error {
  constructor(
    message: string,
    readonly code: "forbidden" | "not_found" | "org_not_active" = "forbidden",
  ) {
    super(message);
    this.name = "AuthzError";
  }
}

export function isPlatformAdmin(actor: Pick<Actor, "userRole">): boolean {
  return actor.userRole === "admin";
}

export function assertPlatformAdmin(actor: Pick<Actor, "userRole">): void {
  if (!isPlatformAdmin(actor)) throw new AuthzError("Admins only.");
}

export function isOrgActive(ctx: Pick<OrgContext, "orgStatus">): boolean {
  return ctx.orgStatus === "active";
}

export function assertOrgActive(ctx: Pick<OrgContext, "orgStatus">): void {
  if (!isOrgActive(ctx)) {
    throw new AuthzError(
      ctx.orgStatus === "pending_approval"
        ? "Your account is waiting for approval."
        : `This organization is ${ctx.orgStatus}.`,
      "org_not_active",
    );
  }
}

/** Any member of an active org can create keys and use the playground. */
export function canCreateKeys(ctx: Pick<OrgContext, "orgStatus" | "orgRole">): boolean {
  return isOrgActive(ctx);
}

/** Owners and org admins revoke any key of the org; members only keys they created. */
export function canRevokeKey(
  ctx: Pick<OrgContext, "orgRole" | "userId">,
  key: { createdBy: string | null },
): boolean {
  if (ctx.orgRole === "owner" || ctx.orgRole === "admin") return true;
  return key.createdBy !== null && key.createdBy === ctx.userId;
}

/** Buying credits and changing auto-recharge: owners and org admins of an active org. */
export function canManageBilling(ctx: Pick<OrgContext, "orgStatus" | "orgRole">): boolean {
  return isOrgActive(ctx) && (ctx.orgRole === "owner" || ctx.orgRole === "admin");
}

/** Emails granted the admin role at sign-in (local dev bootstrap, contracts §4). */
export function parseAdminEmails(raw: string | undefined): Set<string> {
  return new Set(
    (raw ?? "")
      .split(/[,\s]+/)
      .map((e) => e.trim().toLowerCase())
      .filter(Boolean),
  );
}

export function isBootstrapAdmin(email: string | null | undefined, raw: string | undefined) {
  return !!email && parseAdminEmails(raw).has(email.trim().toLowerCase());
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Ids from forms and URLs are untrusted; a non-uuid is "not found", never a SQL error. */
export function isUuid(v: unknown): v is string {
  return typeof v === "string" && UUID_RE.test(v);
}
