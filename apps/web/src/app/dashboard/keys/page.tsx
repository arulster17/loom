import type { Metadata } from "next";

import { Badge, Card, Empty, PageHeader, formatDate, td, tdNum, th, thNum } from "@/components/ui";
import { canRevokeKey } from "@/lib/authz";
import { listKeys } from "@/lib/data/keys";
import { formatUsd } from "@/lib/money";
import { requireActiveOrg } from "@/lib/session";

import { CreateKeyForm, RevokeKeyButton } from "./key-forms";

export const metadata: Metadata = { title: "API keys" };

export default async function KeysPage() {
  const ctx = await requireActiveOrg();
  const keys = await listKeys(ctx);

  return (
    <>
      <PageHeader title="API keys">
        Keys authenticate your API calls. Loom stores only a hash, so a key is shown once, when you create it. A spend
        limit stops a key once it has spent that much in total.
      </PageHeader>
      <Card title="Create a key">
        <CreateKeyForm />
      </Card>
      <Card title="Your keys" className="mt-6">
        {keys.length === 0 ? (
          <Empty>No keys yet.</Empty>
        ) : (
          <div className="-mx-5 overflow-x-auto">
            <table className="min-w-full divide-y divide-zinc-200">
              <thead>
                <tr>
                  <th className={th}>Name</th>
                  <th className={th}>Key</th>
                  <th className={th}>Created</th>
                  <th className={thNum}>Spent</th>
                  <th className={thNum}>Limit</th>
                  <th className={th}>Status</th>
                  <th className={th} />
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {keys.map((k) => (
                  <tr key={k.id} className={k.revokedAt ? "text-zinc-400" : ""}>
                    <td className={td}>{k.name || <span className="text-zinc-400">unnamed</span>}</td>
                    <td className={`${td} font-mono text-xs`}>{k.prefix}…</td>
                    <td className={td}>
                      {formatDate(k.createdAt)}
                      {k.createdByEmail ? <div className="text-xs text-zinc-500">{k.createdByEmail}</div> : null}
                    </td>
                    <td className={tdNum}>{formatUsd(k.spentMicros, 4)}</td>
                    <td className={tdNum}>{k.spendLimitMicros ? formatUsd(k.spendLimitMicros) : "—"}</td>
                    <td className={td}>
                      {k.revokedAt ? <Badge>revoked {formatDate(k.revokedAt).slice(0, 10)}</Badge> : <Badge tone="ok">active</Badge>}
                    </td>
                    <td className={`${td} text-right`}>
                      {!k.revokedAt && canRevokeKey(ctx, k) ? (
                        <RevokeKeyButton keyId={k.id} keyLabel={k.name || k.prefix} />
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}
