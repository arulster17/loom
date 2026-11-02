import type { Metadata } from "next";
import Link from "next/link";

import { Badge, Card, Empty, PageHeader, formatDate, statusLabel, statusTone, td, tdNum, th, thNum } from "@/components/ui";
import { listOrgs, recentAudit } from "@/lib/data/admin";
import { listCatalog } from "@/lib/models";
import { formatUsd } from "@/lib/money";
import { requireAdmin } from "@/lib/session";

import { ApproveRejectButtons } from "./admin-forms";

export const metadata: Metadata = { title: "Admin" };

export default async function AdminPage() {
  const actor = await requireAdmin();
  const [pending, orgs, audit, models] = await Promise.all([
    listOrgs(actor, "pending_approval"),
    listOrgs(actor),
    recentAudit(actor, 30),
    listCatalog(),
  ]);

  return (
    <>
      <PageHeader title="Admin">Approve signups, look after orgs, and see what admins have done.</PageHeader>

      <Card title={`Waiting for approval (${pending.length})`}>
        {pending.length === 0 ? (
          <Empty>No signups waiting.</Empty>
        ) : (
          <div className="-mx-5 overflow-x-auto">
            <table className="min-w-full divide-y divide-zinc-200">
              <thead>
                <tr>
                  <th className={th}>Email</th>
                  <th className={th}>Org</th>
                  <th className={th}>Signed up</th>
                  <th className={th} />
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {pending.map((o) => (
                  <tr key={o.id}>
                    <td className={td}>{o.ownerEmail}</td>
                    <td className={td}>
                      <Link href={`/admin/orgs/${o.id}`} className="underline">
                        {o.name}
                      </Link>
                    </td>
                    <td className={td}>{formatDate(o.createdAt)}</td>
                    <td className={td}>
                      <ApproveRejectButtons orgId={o.id} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="mt-3 text-xs text-zinc-500">Approving activates the org and adds the free signup credit once.</p>
      </Card>

      <Card title="All orgs" className="mt-6">
        <div className="-mx-5 overflow-x-auto">
          <table className="min-w-full divide-y divide-zinc-200">
            <thead>
              <tr>
                <th className={th}>Org</th>
                <th className={th}>Owner</th>
                <th className={th}>Status</th>
                <th className={thNum}>Balance</th>
                <th className={thNum}>Keys</th>
                <th className={th}>Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {orgs.map((o) => (
                <tr key={o.id}>
                  <td className={td}>
                    <Link href={`/admin/orgs/${o.id}`} className="underline">
                      {o.name}
                    </Link>
                  </td>
                  <td className={td}>{o.ownerEmail}</td>
                  <td className={td}>
                    <Badge tone={statusTone[o.status]}>{statusLabel(o.status)}</Badge>
                  </td>
                  <td className={tdNum}>{formatUsd(o.balanceMicros)}</td>
                  <td className={tdNum}>{o.activeKeys}</td>
                  <td className={td}>{formatDate(o.createdAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Models" className="mt-6">
        <ul className="space-y-1 text-sm">
          {models.map((m) => (
            <li key={m.id}>
              <span className="font-mono">{m.id}</span> <span className="text-zinc-500">({m.status})</span>
            </li>
          ))}
        </ul>
        <p className="mt-3 text-xs text-zinc-500">
          To disable a model, set its status to <code>disabled</code> in config/models.yaml and restart the gateway; it
          then answers 404 for that model. A database switch with an audit row is not in the M1 contract yet.
        </p>
      </Card>

      <Card title="Recent admin actions" className="mt-6">
        {audit.length === 0 ? (
          <Empty>None yet.</Empty>
        ) : (
          <ul className="divide-y divide-zinc-100 text-sm">
            {audit.map((a) => (
              <li key={a.id} className="py-2">
                <span className="text-zinc-500">{formatDate(a.createdAt)}</span> · {a.actorEmail ?? "unknown"} ·{" "}
                <span className="font-mono">{a.action}</span> · {a.targetType}{" "}
                {a.targetType === "org" ? (
                  <Link href={`/admin/orgs/${a.targetId}`} className="font-mono text-xs underline">
                    {a.targetId.slice(0, 8)}
                  </Link>
                ) : (
                  <span className="font-mono text-xs">{a.targetId.slice(0, 8)}</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </Card>
    </>
  );
}
