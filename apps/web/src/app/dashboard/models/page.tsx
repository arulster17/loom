import type { Metadata } from "next";

import { Badge, Card, PageHeader, formatInt, td, tdNum, th, thNum } from "@/components/ui";
import { poolLabel } from "@/lib/model-catalog";
import { loadCatalog } from "@/lib/models";
import { formatPricePerMillion } from "@/lib/money";
import { requireActiveOrg } from "@/lib/session";

export const metadata: Metadata = { title: "Models" };

export default async function ModelsPage() {
  await requireActiveOrg();
  const catalog = await loadCatalog();

  return (
    <>
      <PageHeader title="Models">
        Prices are per 1 million tokens, input and output priced separately. Models start on demand: when one is asleep or
        starting, requests get a “temporarily unavailable” reply (HTTP 503 with a Retry-After) instead of waiting.
      </PageHeader>
      {catalog.source !== "gateway" ? (
        <p className="mb-4 text-xs text-zinc-500">The API is not reachable right now; showing the catalog without live status.</p>
      ) : null}
      <Card>
        <div className="-mx-5 overflow-x-auto">
          <table className="min-w-full divide-y divide-zinc-200">
            <thead>
              <tr>
                <th className={th}>Model</th>
                <th className={thNum}>Context</th>
                <th className={thNum}>Input / 1M</th>
                <th className={thNum}>Output / 1M</th>
                <th className={th}>Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {catalog.entries.map(({ model: m, pool }) => {
                const p = poolLabel(pool);
                return (
                  <tr key={m.id}>
                    <td className={td}>
                      <div className="font-medium">{m.displayName}</div>
                      <div className="font-mono text-xs text-zinc-500">{m.id}</div>
                      {m.status === "preview" ? <div className="mt-1 text-xs text-zinc-500">Preview</div> : null}
                    </td>
                    <td className={tdNum}>{formatInt(m.maxContext)} tokens</td>
                    <td className={tdNum}>{formatPricePerMillion(m.inputPerMtokMicros)}</td>
                    <td className={tdNum}>{formatPricePerMillion(m.outputPerMtokMicros)}</td>
                    <td className={td}>
                      <Badge tone={p.tone}>{p.label}</Badge>
                      <div className="mt-1 text-xs text-zinc-500">{p.detail}</div>
                      {m.startupNote ? <div className="mt-1 text-xs text-zinc-500">{m.startupNote}</div> : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
