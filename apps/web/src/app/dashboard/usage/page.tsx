import type { Metadata } from "next";

import { Card, Empty, PageHeader, btn, btnSecondary, formatInt, input, label, td, tdNum, th, thNum } from "@/components/ui";
import { usageRows } from "@/lib/data/usage";
import { formatUsd } from "@/lib/money";
import { requireActiveOrg } from "@/lib/session";
import { groupUsage, parseDayRange, totalsOf, type UsageGroup } from "@/lib/usage";

export const metadata: Metadata = { title: "Usage" };

function GroupTable({ title, first, groups }: { title: string; first: string; groups: UsageGroup[] }) {
  return (
    <Card title={title} className="mt-6">
      {groups.length === 0 ? (
        <Empty>No usage in this period.</Empty>
      ) : (
        <div className="-mx-5 overflow-x-auto">
          <table className="min-w-full divide-y divide-zinc-200">
            <thead>
              <tr>
                <th className={th}>{first}</th>
                <th className={thNum}>Requests</th>
                <th className={thNum}>Input tokens</th>
                <th className={thNum}>Output tokens</th>
                <th className={thNum}>Cost</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {groups.map((g) => (
                <tr key={g.key}>
                  <td className={td}>{g.label}</td>
                  <td className={tdNum}>{formatInt(g.requests)}</td>
                  <td className={tdNum}>{formatInt(g.promptTokens)}</td>
                  <td className={tdNum}>{formatInt(g.completionTokens)}</td>
                  <td className={tdNum}>{formatUsd(g.costMicros, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

export default async function UsagePage(props: PageProps<"/dashboard/usage">) {
  const ctx = await requireActiveOrg();
  const range = parseDayRange(await props.searchParams);
  const rows = await usageRows(ctx, range);
  const total = totalsOf(rows);
  const csvHref = `/api/usage.csv?from=${range.from}&to=${range.to}`;

  return (
    <>
      <PageHeader title="Usage">
        Tokens and cost per day, model and key, in UTC. Includes requests from the last few seconds that are not yet
        on your billing history.
      </PageHeader>
      <form className="flex flex-wrap items-end gap-3">
        <div>
          <label htmlFor="from" className={label}>
            From
          </label>
          <input id="from" name="from" type="date" defaultValue={range.from} className={`${input} mt-1`} />
        </div>
        <div>
          <label htmlFor="to" className={label}>
            To
          </label>
          <input id="to" name="to" type="date" defaultValue={range.to} className={`${input} mt-1`} />
        </div>
        <button type="submit" className={btn}>
          Show
        </button>
        <a href={csvHref} className={btnSecondary} download>
          Download CSV
        </a>
      </form>
      <div className="mt-6 grid gap-4 sm:grid-cols-4">
        <Card title="Cost">
          <p className="text-xl font-semibold tabular-nums">{formatUsd(total.costMicros, 4)}</p>
        </Card>
        <Card title="Requests">
          <p className="text-xl font-semibold tabular-nums">{formatInt(total.requests)}</p>
        </Card>
        <Card title="Input tokens">
          <p className="text-xl font-semibold tabular-nums">{formatInt(total.promptTokens)}</p>
          {total.cachedTokens ? <p className="text-xs text-zinc-500">{formatInt(total.cachedTokens)} cached</p> : null}
        </Card>
        <Card title="Output tokens">
          <p className="text-xl font-semibold tabular-nums">{formatInt(total.completionTokens)}</p>
        </Card>
      </div>
      <GroupTable title="By day" first="Day (UTC)" groups={groupUsage(rows, "day")} />
      <GroupTable title="By model" first="Model" groups={groupUsage(rows, "model")} />
      <GroupTable title="By key" first="Key" groups={groupUsage(rows, "key")} />
    </>
  );
}
