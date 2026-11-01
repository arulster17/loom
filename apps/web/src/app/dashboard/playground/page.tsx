import type { Metadata } from "next";

import { PageHeader } from "@/components/ui";
import { getPlaygroundKey } from "@/lib/data/playground";
import { listCatalog } from "@/lib/models";
import { gatewayModelIds, publicApiBase } from "@/lib/services";
import { requireActiveOrg } from "@/lib/session";

import { Playground, type PlaygroundModel } from "./playground";

export const metadata: Metadata = { title: "Playground" };

export default async function PlaygroundPage() {
  const ctx = await requireActiveOrg();
  const catalog = await listCatalog();
  let ids: string[];
  let gatewayDown = false;
  try {
    ids = await gatewayModelIds(await getPlaygroundKey(ctx)); // the dropdown is what the API offers
  } catch {
    ids = catalog.map((m) => m.id);
    gatewayDown = true;
  }
  const models: PlaygroundModel[] = ids.map((id) => {
    const info = catalog.find((m) => m.id === id);
    return { id, label: info?.displayName ?? id, maxContext: info?.maxContext ?? 32_768, note: info?.startupNote ?? null };
  });

  return (
    <>
      <PageHeader title="Playground">
        Chat with a model using your org&apos;s credit. Requests here are billed like API calls and appear in Usage as
        “Playground”. Nothing you type is stored.
      </PageHeader>
      <Playground models={models} apiBase={publicApiBase()} gatewayDown={gatewayDown} />
    </>
  );
}
