import type { Metadata } from "next";
import Link from "next/link";

import { Card, PageHeader, formatInt } from "@/components/ui";
import { getBalance } from "@/lib/data/billing";
import { listKeys } from "@/lib/data/keys";
import { usageRows } from "@/lib/data/usage";
import { formatUsd } from "@/lib/money";
import { publicApiBase } from "@/lib/services";
import { requireActiveOrg } from "@/lib/session";
import { parseDayRange, totalsOf } from "@/lib/usage";

export const metadata: Metadata = { title: "Overview" };

export default async function OverviewPage() {
  const ctx = await requireActiveOrg();
  const [balance, keys, rows] = await Promise.all([
    getBalance(ctx),
    listKeys(ctx),
    usageRows(ctx, parseDayRange({})),
  ]);
  const month = totalsOf(rows);
  const activeKeys = keys.filter((k) => !k.revokedAt).length;

  return (
    <>
      <PageHeader title={ctx.orgName}>Your balance, keys and the last 30 days at a glance.</PageHeader>
      <div className="grid gap-4 sm:grid-cols-3">
        <Card title="Balance">
          <p className="text-2xl font-semibold tabular-nums">{formatUsd(balance.balanceMicros)}</p>
          <Link href="/dashboard/billing" className="mt-2 inline-block text-sm text-zinc-600 underline">
            Add credit
          </Link>
        </Card>
        <Card title="Last 30 days">
          <p className="text-2xl font-semibold tabular-nums">{formatUsd(month.costMicros)}</p>
          <p className="mt-2 text-sm text-zinc-600">
            {formatInt(month.requests)} requests, {formatInt(month.promptTokens + month.completionTokens)} tokens
          </p>
        </Card>
        <Card title="API keys">
          <p className="text-2xl font-semibold tabular-nums">{activeKeys}</p>
          <Link href="/dashboard/keys" className="mt-2 inline-block text-sm text-zinc-600 underline">
            Manage keys
          </Link>
        </Card>
      </div>
      <Card title="Quick start" className="mt-6">
        <p className="text-sm text-zinc-700">
          Loom works with the OpenAI SDKs: change the base URL and use a Loom key. Try models in the{" "}
          <Link href="/dashboard/playground" className="underline">
            playground
          </Link>{" "}
          first, or call the API directly:
        </p>
        <pre className="mt-3 overflow-x-auto rounded-md bg-zinc-900 p-3 text-xs text-zinc-100">
          {`curl ${publicApiBase()}/chat/completions \\
  -H "Authorization: Bearer $LOOM_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "qwen3-8b", "messages": [{"role": "user", "content": "Hello"}]}'`}
        </pre>
        <p className="mt-3 text-xs text-zinc-500">
          Every response carries its cost in the X-Loom-Cost-USD header. Loom does not store your prompts or completions.
        </p>
      </Card>
    </>
  );
}
