import type { Metadata } from "next";

import { PageHeader } from "@/components/ui";
import { loadCatalog } from "@/lib/models";
import { publicApiBase } from "@/lib/services";
import { requireActiveOrg } from "@/lib/session";

import { Playground, type PlaygroundModel } from "./playground";

export const metadata: Metadata = { title: "Playground" };

export default async function PlaygroundPage() {
  await requireActiveOrg();
  const catalog = await loadCatalog(); // the dropdown is the gateway's GET /v1/models
  const models: PlaygroundModel[] = catalog.entries.map(({ model, pool }) => ({
    id: model.id,
    label: pool.state === "ready" || pool.state === "unknown" ? model.displayName : `${model.displayName} (not running)`,
    maxContext: model.maxContext,
    note: model.startupNote,
  }));

  return (
    <>
      <PageHeader title="Playground">
        Chat with a model using your org&apos;s credit. Requests here are billed like API calls and appear in Usage as
        “Playground”. Nothing you type is stored.
      </PageHeader>
      <Playground models={models} apiBase={publicApiBase()} gatewayDown={catalog.source !== "gateway"} />
    </>
  );
}
