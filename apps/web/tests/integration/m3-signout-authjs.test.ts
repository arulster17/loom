// W3 (R13), docs/m3-review-fixes-spec.md: every sign-out revokes the playground keys.
// POST /api/auth/signout, through @auth/core with the app's real config and database
// sessions, triggers Auth.js's `events.signOut`; the user's playground keys must be
// revoked as the app's own sign-out button does.

import { randomUUID } from "node:crypto";

import { beforeAll, describe, expect, it } from "vitest";

import type { Actor, OrgContext } from "@/lib/authz";
import { getPool } from "@/lib/db";
import { getPlaygroundKey } from "@/lib/data/playground";

import { activeUser, adapter, hasDb, makeAdmin } from "./helpers";
import { Browser } from "./m3-helpers";

/** A browser holding a database session for the user (the cookie Auth.js sets at sign-in). */
async function signedIn(ctx: OrgContext): Promise<{ browser: Browser; sessionToken: string }> {
  const sessionToken = randomUUID();
  await adapter().createSession!({ sessionToken, userId: ctx.userId, expires: new Date(Date.now() + 3600_000) });
  const browser = new Browser();
  browser.cookies.set("authjs.session-token", sessionToken);
  return { browser, sessionToken };
}

async function livePlaygroundKeys(userId: string): Promise<number> {
  const r = await getPool().query(
    `SELECT count(*)::int AS n FROM api_keys
     WHERE created_by = $1 AND 'playground' = ANY (scopes) AND revoked_at IS NULL`,
    [userId],
  );
  return r.rows[0].n;
}

describe.skipIf(!hasDb)("W3: Auth.js's sign-out revokes the playground keys", () => {
  let admin: Actor;

  beforeAll(async () => {
    admin = await makeAdmin();
  });

  it("W3: POST /api/auth/signout revokes the user's playground keys and ends the session", async () => {
    const u = await activeUser(admin);
    await getPlaygroundKey(u);
    expect(await livePlaygroundKeys(u.userId)).toBe(1);
    const { browser, sessionToken } = await signedIn(u);
    // Auth.js sees the session before signing out.
    const before = await browser.request("GET", "/api/auth/session");
    expect(((await before.json()) as { user?: { id?: string } } | null)?.user?.id).toBe(u.userId);

    const res = await browser.postSignOut();
    expect(res.status).toBeLessThan(400);

    expect(await adapter().getSessionAndUser!(sessionToken)).toBeNull();
    expect(await livePlaygroundKeys(u.userId)).toBe(0);
  });

  it("W3: signing out revokes only that user's playground keys", async () => {
    const u = await activeUser(admin);
    const other = await activeUser(admin);
    await getPlaygroundKey(u);
    await getPlaygroundKey(other);
    const { browser } = await signedIn(u);

    await browser.postSignOut();

    expect(await livePlaygroundKeys(u.userId)).toBe(0);
    expect(await livePlaygroundKeys(other.userId)).toBe(1);
  });

  it("W3: a user's next playground key after signing out is a new one", async () => {
    const u = await activeUser(admin);
    const k1 = await getPlaygroundKey(u);
    const { browser } = await signedIn(u);
    await browser.postSignOut();
    const k2 = await getPlaygroundKey(u);
    expect(k2).not.toBe(k1);
    expect(await livePlaygroundKeys(u.userId)).toBe(1);
  });
});
