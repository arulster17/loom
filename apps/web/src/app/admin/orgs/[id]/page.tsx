import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { Badge, Card, Empty, PageHeader, formatDate, statusLabel, statusTone } from "@/components/ui";
import { getOrgDetail } from "@/lib/data/admin";
import { formatUsd } from "@/lib/money";
import { requireAdmin } from "@/lib/session";

import { AdjustCreditsForm, ApproveRejectButtons, SuspendForm } from "../../admin-forms";

export const metadata: Metadata = { title: "Org" };

export default async function OrgAdminPage(props: PageProps<"/admin/orgs/[id]">) {
  const actor = await requireAdmin();
  const { id } = await props.params;
  const org = await getOrgDetail(actor, id);
  if (!org) notFound();

  return (
    <>
      <Link href="/admin" className="text-sm text-zinc-600 underline">
        ← All orgs
      </Link>
      <div className="mt-2">
        <PageHeader title={org.name}>
          <span className="font-mono text-xs">{org.id}</span>
        </PageHeader>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <Card title="Status">
          <Badge tone={statusTone[org.status]}>{statusLabel(org.status)}</Badge>
          {org.suspendedReason ? <p className="mt-2 text-sm text-zinc-600">Reason: {org.suspendedReason}</p> : null}
          <p className="mt-2 text-xs text-zinc-500">
            Created {formatDate(org.createdAt)}
            {org.approvedAt ? `, approved ${formatDate(org.approvedAt)}` : ""}
          </p>
        </Card>
        <Card title="Balance">
          <p className="text-2xl font-semibold tabular-nums">{formatUsd(org.balanceMicros)}</p>
          <p className="mt-1 text-xs text-zinc-500">{org.activeKeys} active API keys</p>
        </Card>
        <Card title="Members">
          <ul className="space-y-1 text-sm">
            {org.members.map((m) => (
              <li key={`${m.email}-${m.role}`}>
                {m.email} <span className="text-zinc-500">({m.role})</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      {org.status === "pending_approval" ? (
        <Card title="Signup" className="mt-6">
          <ApproveRejectButtons orgId={org.id} />
        </Card>
      ) : null}

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <Card title="Suspend">
          <SuspendForm orgId={org.id} status={org.status} />
        </Card>
        <Card title="Adjust credit">
          <AdjustCreditsForm orgId={org.id} />
        </Card>
      </div>

      <Card title="Audit log for this org" className="mt-6">
        {org.audit.length === 0 ? (
          <Empty>No admin actions yet.</Empty>
        ) : (
          <ul className="divide-y divide-zinc-100 text-sm">
            {org.audit.map((a) => (
              <li key={a.id} className="py-2">
                <span className="text-zinc-500">{formatDate(a.createdAt)}</span> · {a.actorEmail ?? "unknown"} ·{" "}
                <span className="font-mono">{a.action}</span>{" "}
                <span className="font-mono text-xs text-zinc-500">{JSON.stringify(a.details)}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </>
  );
}
