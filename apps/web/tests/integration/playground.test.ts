import { beforeAll, describe, expect, it } from "vitest";

import { hashApiKey, isWellFormed } from "@/lib/apikeys";
import { AuthzError, type Actor } from "@/lib/authz";
import { getPool } from "@/lib/db";
import { listKeys } from "@/lib/data/keys";
import { getPlaygroundKey, revokePlaygroundKeys } from "@/lib/data/playground";

import { activeUser, hasDb, makeAdmin, signUp } from "./helpers";

async function rowFor(plaintext: string) {
  const r = await getPool().query("SELECT * FROM api_keys WHERE key_hash = $1", [hashApiKey(plaintext)]);
  return r.rows[0];
}

describe.skipIf(!hasDb)("playground session keys", () => {
  let admin: Actor;
  beforeAll(async () => {
    admin = await makeAdmin();
  });

  it("mints a well-formed key for the caller's org, stores only its hash, and reuses it", async () => {
    const u = await activeUser(admin);
    const k1 = await getPlaygroundKey(u);
    expect(isWellFormed(k1)).toBe(true);
    const row = await rowFor(k1);
    expect(row).toMatchObject({ org_id: u.orgId, created_by: u.userId, name: "Playground", scopes: ["playground"], revoked_at: null });
    expect(await getPlaygroundKey(u)).toBe(k1);
    // Hidden from the API keys page.
    expect((await listKeys(u)).map((k) => k.id)).not.toContain(row.id);
  });

  it("is refused for an org that is not active", async () => {
    const pending = await signUp();
    await expect(getPlaygroundKey(pending)).rejects.toBeInstanceOf(AuthzError);
  });

  it("rotates after the TTL: the old key is revoked, a new one minted", async () => {
    const u = await activeUser(admin);
    const old = await getPlaygroundKey(u);
    await getPool().query("UPDATE api_keys SET created_at = now() - interval '2 hours' WHERE key_hash = $1", [hashApiKey(old)]);
    const fresh = await getPlaygroundKey(u);
    expect(fresh).not.toBe(old);
    expect((await rowFor(old)).revoked_at).not.toBeNull();
    expect((await rowFor(fresh)).revoked_at).toBeNull();
  });

  it("is per user and per org, and sign-out revokes only that user's keys", async () => {
    const u = await activeUser(admin);
    const v = await activeUser(admin);
    const ku = await getPlaygroundKey(u);
    const kv = await getPlaygroundKey(v);
    expect(ku).not.toBe(kv);
    expect((await rowFor(kv)).org_id).toBe(v.orgId);
    expect(await revokePlaygroundKeys(u.userId)).toBe(1);
    expect((await rowFor(ku)).revoked_at).not.toBeNull();
    expect((await rowFor(kv)).revoked_at).toBeNull();
  });
});
