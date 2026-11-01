import { beforeAll, describe, expect, it } from "vitest";

import { AuthzError, type Actor, type OrgContext } from "@/lib/authz";
import { getPool } from "@/lib/db";
import { creditHistory, getAutoRecharge, getBalance, saveAutoRecharge } from "@/lib/data/billing";
import { createKey } from "@/lib/data/keys";
import { usageRows } from "@/lib/data/usage";
import { parseDayRange, totalsOf, usageCsv } from "@/lib/usage";

import { activeUser, addMember, hasDb, insertUsage, makeAdmin } from "./helpers";

describe.skipIf(!hasDb)("usage and billing are org-scoped", () => {
  let admin: Actor;
  let a: OrgContext;
  let b: OrgContext;
  let keyA: string;
  let keyB: string;

  beforeAll(async () => {
    admin = await makeAdmin();
    a = await activeUser(admin);
    b = await activeUser(admin);
    keyA = (await createKey(a, { name: "a", spendLimitMicros: null })).id;
    keyB = (await createKey(b, { name: "b", spendLimitMicros: null })).id;
    await insertUsage(a, keyA, { costMicros: 1200, model: "qwen3-8b" });
    await insertUsage(a, keyA, { costMicros: 800, model: "llama-3.3-70b-instruct-fp8" });
    await insertUsage(b, keyB, { costMicros: 999_999 });
  });

  it("shows an org only its own usage", async () => {
    const rows = await usageRows(a, parseDayRange({}));
    expect(rows.every((r) => r.apiKeyId === keyA)).toBe(true);
    expect(totalsOf(rows)).toMatchObject({ requests: 2, costMicros: 2000n });
    const rowsB = await usageRows(b, parseDayRange({}));
    expect(rowsB.map((r) => r.apiKeyId)).toEqual([keyB]);
    expect(usageCsv(rows)).not.toContain(keyB);
  });

  it("ignores usage outside the date range", async () => {
    const yesterday = new Date(Date.now() - 86_400_000).toISOString().slice(0, 10);
    const rows = await usageRows(a, parseDayRange({ from: "2020-01-01", to: yesterday }));
    expect(rows).toEqual([]);
  });

  it("balance is credit minus all metered usage", async () => {
    expect((await getBalance(a)).balanceMicros).toBe(5_000_000n - 2000n);
    expect((await getBalance(b)).balanceMicros).toBe(5_000_000n - 999_999n);
  });

  it("credit history shows the org's own credits only", async () => {
    const hist = await creditHistory(a);
    expect(hist).toHaveLength(1);
    expect(hist[0]).toMatchObject({ kind: "promo", amountMicros: 5_000_000n });
    const ids = (await creditHistory(b)).map((h) => h.id);
    expect(ids).not.toContain(hist[0].id);
  });

  it("auto-recharge: owners save it, members cannot, other orgs are untouched", async () => {
    await saveAutoRecharge(a, { enabled: true, thresholdMicros: 2_000_000n, amountMicros: 10_000_000n });
    expect(await getAutoRecharge(a)).toEqual({ enabled: true, thresholdMicros: 2_000_000n, amountMicros: 10_000_000n });
    expect((await getAutoRecharge(b)).enabled).toBe(false);
    const member = await addMember(a.orgId, "member");
    await expect(
      saveAutoRecharge(member, { enabled: false, thresholdMicros: 0n, amountMicros: 10_000_000n }),
    ).rejects.toBeInstanceOf(AuthzError);
    await expect(
      saveAutoRecharge(a, { enabled: true, thresholdMicros: 0n, amountMicros: 1_000_000n }),
    ).rejects.toBeInstanceOf(RangeError);
    const n = await getPool().query("SELECT count(*)::int AS n FROM auto_recharge WHERE org_id = $1", [b.orgId]);
    expect(n.rows[0].n).toBe(0);
  });
});
