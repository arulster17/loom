// W4 (R14), docs/m3-review-fixes-spec.md: a key's spend limit has an upper bound of
// $1,000,000 (10^12 micros). Above it the API keys form's action and the data layer
// refuse with a message naming the bound; exactly the bound is accepted; no input gives
// "Something went wrong".

import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { Actor, OrgContext } from "@/lib/authz";
import { getPool } from "@/lib/db";
import { createKey } from "@/lib/data/keys";

import { activeUser, hasDb, makeAdmin } from "./helpers";
import { form } from "./m3-helpers";

const session = vi.hoisted(() => ({ userId: null as string | null }));

vi.mock("@/auth", () => ({
  auth: vi.fn(async () => (session.userId ? { user: { id: session.userId } } : null)),
  signIn: vi.fn(async () => undefined),
  signOut: vi.fn(async () => undefined),
}));
vi.mock("next-auth", () => ({ AuthError: class AuthError extends Error {} }));
vi.mock("next/cache", () => ({ revalidatePath: vi.fn() }));

const { createKeyAction } = await import("@/app/dashboard/keys/actions");

const BOUND = 1_000_000_000_000n; // $1,000,000 in micros
const NAMES_BOUND = /\$1,000,000(?![\d,])/; // "$1,000,000" or "$1,000,000.00"

async function keyCount(orgId: string): Promise<number> {
  const r = await getPool().query("SELECT count(*)::int AS n FROM api_keys WHERE org_id = $1", [orgId]);
  return r.rows[0].n;
}

async function storedLimit(orgId: string, prefix: string): Promise<string | null> {
  const r = await getPool().query("SELECT spend_limit_micros FROM api_keys WHERE org_id = $1 AND prefix = $2", [
    orgId,
    prefix,
  ]);
  return r.rows[0]?.spend_limit_micros ?? null;
}

describe.skipIf(!hasDb)("W4: a key's spend limit has an upper bound", () => {
  let admin: Actor;
  let ctx: OrgContext;

  beforeAll(async () => {
    admin = await makeAdmin();
  });

  beforeEach(async () => {
    ctx = await activeUser(admin);
    session.userId = ctx.userId;
  });

  it("W4: the action accepts a spend limit of exactly $1,000,000", async () => {
    for (const typed of ["1000000", "$1,000,000.00", "1000000.000000"]) {
      const r = await createKeyAction(undefined, form({ name: `max ${typed}`, spend_limit: typed }));
      expect(r?.ok, JSON.stringify(r)).toBe(true);
      if (r?.ok) expect(await storedLimit(ctx.orgId, r.prefix)).toBe(BOUND.toString());
    }
  });

  it.each([
    ["one micro over", "1000000.000001"],
    ["one cent over", "1,000,000.01"],
    ["a dollar over", "1000001"],
    ["past Postgres bigint", "10000000000000"],
    ["absurdly large", "99999999999999999999999999999999"],
  ])("W4: the action refuses a spend limit %s with a message naming the bound", async (_label, typed) => {
    const r = await createKeyAction(undefined, form({ name: "too much", spend_limit: typed }));
    expect(r?.ok).toBe(false);
    const error = r && !r.ok ? r.error : "";
    expect(error).toMatch(NAMES_BOUND);
    expect(error).not.toMatch(/something went wrong/i);
    expect(await keyCount(ctx.orgId)).toBe(0);
  });

  it("W4: the data layer accepts exactly $1,000,000", async () => {
    const key = await createKey(ctx, { name: "max", spendLimitMicros: BOUND });
    expect(await storedLimit(ctx.orgId, key.prefix)).toBe(BOUND.toString());
  });

  it.each([
    ["one micro over", BOUND + 1n],
    ["past Postgres bigint", 2n ** 63n],
  ])("W4: the data layer refuses a spend limit %s with a message naming the bound", async (_label, micros) => {
    await expect(createKey(ctx, { name: "too much", spendLimitMicros: micros })).rejects.toThrow(NAMES_BOUND);
    expect(await keyCount(ctx.orgId)).toBe(0);
  });
});
