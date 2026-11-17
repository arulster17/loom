// W2 (R12), docs/m3-review-fixes-spec.md: an admin credit adjustment posts once per form.
// The adjust-credits form carries a per-render `form_token` (a random UUID); the ledger
// idempotency key is `admin:adjust:<token>`. Tested through the action the form posts to
// (adjustCreditsAction) and the admin org page that renders the form; only the identity
// layer (`auth()` from @/auth) is faked.

import { randomUUID } from "node:crypto";

import { renderToString } from "react-dom/server";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { Actor, OrgContext } from "@/lib/authz";
import { getPool } from "@/lib/db";

import { activeUser, hasDb, makeAdmin } from "./helpers";
import { adminSnapshot, creditMicros, form } from "./m3-helpers";

const session = vi.hoisted(() => ({ userId: null as string | null }));

vi.mock("@/auth", () => ({
  auth: vi.fn(async () => (session.userId ? { user: { id: session.userId } } : null)),
  signIn: vi.fn(async () => undefined),
  signOut: vi.fn(async () => undefined),
}));
vi.mock("next-auth", () => ({ AuthError: class AuthError extends Error {} }));
vi.mock("next/cache", () => ({ revalidatePath: vi.fn() }));

const { adjustCreditsAction } = await import("@/app/admin/actions");
const { default: OrgAdminPage } = await import("@/app/admin/orgs/[id]/page");

async function ledgerFor(token: string): Promise<number> {
  const r = await getPool().query("SELECT count(*)::int AS n FROM ledger_transactions WHERE idempotency_key = $1", [
    `admin:adjust:${token}`,
  ]);
  return r.rows[0].n;
}

async function adjustAudits(orgId: string): Promise<number> {
  const r = await getPool().query(
    "SELECT count(*)::int AS n FROM admin_audit_log WHERE action = 'org.credit_adjust' AND target_type = 'org' AND target_id = $1",
    [orgId],
  );
  return r.rows[0].n;
}

/** The `value` of every `<input name="form_token">` in the HTML. */
function formTokens(html: string): string[] {
  return [...html.matchAll(/<input\b[^>]*>/g)]
    .map((m) => m[0])
    .filter((tag) => /\bname="form_token"/.test(tag))
    .map((tag) => /\bvalue="([^"]*)"/.exec(tag)?.[1] ?? "");
}

/** The admin org page (/admin/orgs/<id>), rendered to HTML as the server sends it. */
async function renderOrgPage(orgId: string): Promise<string> {
  const page = await OrgAdminPage({ params: Promise.resolve({ id: orgId }) } as never);
  return renderToString(page as Parameters<typeof renderToString>[0]);
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

describe.skipIf(!hasDb)("W2: an admin credit adjustment posts once per form", () => {
  let admin: Actor;
  let org: OrgContext;

  beforeAll(async () => {
    admin = await makeAdmin();
  });

  beforeEach(async () => {
    org = await activeUser(admin);
    session.userId = admin.userId;
  });

  it("W2: a resubmit with the same form_token posts one ledger transaction and one audit row, and answers the same success", async () => {
    const token = randomUUID();
    const before = await creditMicros(org.orgId);
    const fields = { org_id: org.orgId, amount: "7", memo: "goodwill", form_token: token };
    const first = await adjustCreditsAction(undefined, form(fields));
    expect(first?.error).toBeUndefined();
    expect(first?.ok).toBeTruthy();
    const second = await adjustCreditsAction(undefined, form(fields));
    expect(second).toEqual(first);

    expect(await ledgerFor(token)).toBe(1);
    expect(await adjustAudits(org.orgId)).toBe(1);
    expect(await creditMicros(org.orgId)).toBe(before + 7_000_000n);
  });

  it("W2: the same form posted from two tabs at once posts once", async () => {
    const token = randomUUID();
    const before = await creditMicros(org.orgId);
    const fields = { org_id: org.orgId, amount: "-1.50", memo: "correction", form_token: token };
    const results = await Promise.all([
      adjustCreditsAction(undefined, form(fields)),
      adjustCreditsAction(undefined, form(fields)),
    ]);
    for (const r of results) {
      expect(r?.error).toBeUndefined();
      expect(r?.ok).toBeTruthy();
    }
    expect(results[1]).toEqual(results[0]);
    expect(await ledgerFor(token)).toBe(1);
    expect(await adjustAudits(org.orgId)).toBe(1);
    expect(await creditMicros(org.orgId)).toBe(before - 1_500_000n);
  });

  it("W2: two forms (two tokens) post two adjustments", async () => {
    const before = await creditMicros(org.orgId);
    for (const token of [randomUUID(), randomUUID()]) {
      const r = await adjustCreditsAction(
        undefined,
        form({ org_id: org.orgId, amount: "2", memo: "twice", form_token: token }),
      );
      expect(r?.error).toBeUndefined();
      expect(await ledgerFor(token)).toBe(1);
    }
    expect(await adjustAudits(org.orgId)).toBe(2);
    expect(await creditMicros(org.orgId)).toBe(before + 4_000_000n);
  });

  it("W2: refuses a used form_token posted again for another org or another amount, changing nothing", async () => {
    const other = await activeUser(admin);
    const token = randomUUID();
    const fields = { org_id: org.orgId, amount: "5", memo: "once", form_token: token };
    const first = await adjustCreditsAction(undefined, form(fields));
    expect(first?.error).toBeUndefined();
    expect(first?.ok).toBeTruthy();

    const before = await adminSnapshot();
    const credit = await creditMicros(org.orgId);
    const otherCredit = await creditMicros(other.orgId);
    for (const [label, change] of [
      ["another org_id", { org_id: other.orgId }],
      ["another amount", { amount: "6" }],
    ] as const) {
      const r = await adjustCreditsAction(undefined, form({ ...fields, ...change }));
      expect(r?.error, label).toBeTruthy();
      expect(r?.ok, label).toBeUndefined();
      expect(await adminSnapshot(), label).toEqual(before);
    }
    expect(await creditMicros(org.orgId)).toBe(credit);
    expect(await creditMicros(other.orgId)).toBe(otherCredit);
    expect(await ledgerFor(token)).toBe(1);
  });

  it.each([
    ["missing", undefined],
    ["empty", ""],
    ["not a UUID", "not-a-token"],
    ["a UUID with junk around it", ` ${randomUUID()}x`],
    ["an idempotency key, not a token", `admin:adjust:${randomUUID()}`],
  ])("W2: refuses a form_token that is %s, writing nothing", async (_label, token) => {
    const fields: Record<string, string> = { org_id: org.orgId, amount: "3", memo: "no token" };
    if (token !== undefined) fields.form_token = token;
    const before = await adminSnapshot();
    const r = await adjustCreditsAction(undefined, form(fields));
    expect(r?.error).toBeTruthy();
    expect(r?.ok).toBeUndefined();
    expect(await adminSnapshot()).toEqual(before);
  });

  it("W2: each render of the adjust-credits form carries a new random UUID token", async () => {
    const a = formTokens(await renderOrgPage(org.orgId));
    const b = formTokens(await renderOrgPage(org.orgId));
    expect(a).toHaveLength(1);
    expect(b).toHaveLength(1);
    expect(a[0]).toMatch(UUID);
    expect(b[0]).toMatch(UUID);
    expect(a[0]).not.toBe(b[0]);
  });

  it("W2: a token from a rendered form posts once", async () => {
    const [token] = formTokens(await renderOrgPage(org.orgId));
    expect(token).toMatch(UUID);
    const fields = { org_id: org.orgId, amount: "1", memo: "from the page", form_token: token };
    expect((await adjustCreditsAction(undefined, form(fields)))?.ok).toBeTruthy();
    expect((await adjustCreditsAction(undefined, form(fields)))?.ok).toBeTruthy();
    expect(await ledgerFor(token)).toBe(1);
    expect(await adjustAudits(org.orgId)).toBe(1);
  });
});
