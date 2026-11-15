import { execFileSync } from "node:child_process";
import path from "node:path";

import { Client } from "pg";
import { beforeAll, describe, expect, it } from "vitest";

import { hashApiKey } from "@/lib/apikeys";
import { AuthzError, type Actor, type OrgContext } from "@/lib/authz";
import { getPool } from "@/lib/db";
import { createKey, listKeys, revokeKey } from "@/lib/data/keys";

import { activeUser, addMember, hasDb, keyIdByPrefix, makeAdmin, signUp } from "./helpers";

const REPO_ROOT = path.resolve(__dirname, "../../../..");

async function expectAuthz(p: Promise<unknown>, code: AuthzError["code"]) {
  await expect(p).rejects.toSatisfy((e: unknown) => e instanceof AuthzError && e.code === code);
}

describe.skipIf(!hasDb)("API keys", () => {
  let admin: Actor;
  let a: OrgContext;
  let b: OrgContext;

  beforeAll(async () => {
    admin = await makeAdmin();
    a = await activeUser(admin);
    b = await activeUser(admin);
  });

  it("stores only the prefix and the SHA-256, matching the Python implementation", async () => {
    const key = await createKey(a, { name: "prod", spendLimitMicros: 2_500_000n });
    expect(key.plaintext).toMatch(/^loom_sk_[0-9A-Za-z]{12}_[0-9A-Za-z]{40}$/);
    const row = await getPool().query("SELECT prefix, key_hash, spend_limit_micros, name FROM api_keys WHERE id = $1", [key.id]);
    expect(row.rows[0].prefix).toBe(key.plaintext.slice(0, 20));
    expect(Buffer.compare(row.rows[0].key_hash, hashApiKey(key.plaintext))).toBe(0);
    expect(row.rows[0].spend_limit_micros).toBe("2500000");
    // The gateway hashes with loom_platform.apikeys: same bytes.
    const py = execFileSync(
      "uv",
      ["run", "python", "-c", "import sys; from loom_platform.apikeys import hash_api_key; print(hash_api_key(sys.argv[1]).hex())", key.plaintext],
      { cwd: REPO_ROOT, encoding: "utf8" },
    ).trim();
    expect(py).toBe(row.rows[0].key_hash.toString("hex"));
    // The plaintext is nowhere in the database.
    const leak = await getPool().query("SELECT count(*)::int AS n FROM api_keys WHERE name = $1 OR prefix = $1", [key.plaintext]);
    expect(leak.rows[0].n).toBe(0);
  });

  it("refuses keys for an org that is not approved", async () => {
    const pending = await signUp();
    await expectAuthz(createKey(pending, { name: "x", spendLimitMicros: null }), "org_not_active");
  });

  it("re-checks the org status in the database, not the caller's stale context", async () => {
    const c = await activeUser(admin);
    await getPool().query("UPDATE orgs SET status = 'suspended' WHERE id = $1", [c.orgId]);
    await expectAuthz(createKey(c, { name: "x", spendLimitMicros: null }), "org_not_active");
  });

  it("lists only the caller's own org's keys", async () => {
    const ka = await createKey(a, { name: "a-key", spendLimitMicros: null });
    const kb = await createKey(b, { name: "b-key", spendLimitMicros: null });
    const listA = await listKeys(a);
    expect(listA.map((k) => k.id)).toContain(ka.id);
    expect(listA.map((k) => k.id)).not.toContain(kb.id);
    expect((await listKeys(b)).map((k) => k.id)).not.toContain(ka.id);
  });

  it("cannot revoke another org's key: it is simply not found", async () => {
    const kb = await createKey(b, { name: "b-victim", spendLimitMicros: null });
    await expectAuthz(revokeKey(a, kb.id), "not_found");
    const row = await getPool().query("SELECT revoked_at FROM api_keys WHERE id = $1", [kb.id]);
    expect(row.rows[0].revoked_at).toBeNull();
  });

  it("treats a malformed key id as not found", async () => {
    await expectAuthz(revokeKey(a, "'; DROP TABLE api_keys; --"), "not_found");
  });

  it("lets members revoke only their own keys; owners revoke any", async () => {
    const member = await addMember(a.orgId, "member");
    const ownerKey = await createKey(a, { name: "owner's", spendLimitMicros: null });
    const memberKey = await createKey(member, { name: "member's", spendLimitMicros: null });
    await expectAuthz(revokeKey(member, ownerKey.id), "forbidden");
    await revokeKey(member, memberKey.id);
    await revokeKey(a, ownerKey.id);
    const rows = await getPool().query("SELECT id FROM api_keys WHERE id = ANY($1) AND revoked_at IS NOT NULL", [
      [ownerKey.id, memberKey.id],
    ]);
    expect(rows.rowCount).toBe(2);
  });

  it("revocation notifies loom_auth_changed with the key hash, so the gateway drops it at once", async () => {
    const key = await createKey(a, { name: "to-revoke", spendLimitMicros: null });
    const listener = new Client({ connectionString: process.env.DATABASE_URL });
    await listener.connect();
    try {
      await listener.query("LISTEN loom_auth_changed");
      const got = new Promise<Record<string, string>>((resolve) => {
        listener.on("notification", (n) => {
          const p = JSON.parse(n.payload ?? "{}");
          if (p.key_id === key.id) resolve(p);
        });
      });
      await revokeKey(a, key.id);
      const payload = await got;
      expect(payload).toMatchObject({ kind: "api_key", org_id: a.orgId, key_hash: hashApiKey(key.plaintext).toString("hex") });
    } finally {
      await listener.end();
    }
    expect(await keyIdByPrefix(key.prefix)).toBe(key.id);
  });
});
