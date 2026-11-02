import "server-only";

import { readFile } from "node:fs/promises";
import path from "node:path";

import yaml from "js-yaml";

import {
  catalogFromGateway,
  catalogFromRegistry,
  parsePoolHash,
  type CatalogEntry,
  type ModelInfo,
  type PoolStatus,
} from "@/lib/model-catalog";
import { connectedRedis } from "@/lib/redis";
import { gatewayUrl } from "@/lib/services";

/** config/models.yaml: $LOOM_MODELS_YAML, else the repo copy (dev runs from apps/web). */
export function modelsYamlPath(): string {
  return process.env.LOOM_MODELS_YAML ?? path.resolve(process.cwd(), "../../config/models.yaml");
}

let cached: { at: number; models: ModelInfo[] } | undefined;

/** The models the public API offers (priced and not disabled), from the registry. */
export async function listCatalog(): Promise<ModelInfo[]> {
  if (cached && Date.now() - cached.at < 60_000) return cached.models;
  // Runtime path (the Docker image copies the file in); not part of the build trace.
  const doc = yaml.load(await readFile(/*turbopackIgnore: true*/ modelsYamlPath(), "utf8"));
  const models = catalogFromRegistry(doc);
  cached = { at: Date.now(), models };
  return models;
}

/**
 * Pool state per model from `loom:pool:<model>` (contracts §10), read-only. If Redis is
 * unreachable every model reads "unknown"; the page still renders.
 */
export async function poolStatuses(modelIds: string[]): Promise<Record<string, PoolStatus>> {
  const out: Record<string, PoolStatus> = Object.fromEntries(modelIds.map((id) => [id, { state: "unknown" }]));
  try {
    const r = await connectedRedis();
    const results = await Promise.all(modelIds.map((id) => r.hgetall(`loom:pool:${id}`)));
    modelIds.forEach((id, i) => {
      const h = results[i];
      if (!h || Object.keys(h).length === 0) {
        out[id] = { state: "cold" }; // no hash = cold (contracts §10)
        return;
      }
      out[id] = parsePoolHash(h);
    });
  } catch {
    // leave "unknown"
  }
  return out;
}

export interface Catalog {
  entries: CatalogEntry[];
  source: "gateway" | "registry"; // registry = the gateway was unreachable
}

/**
 * What the API offers right now: the gateway's public `GET /v1/models` (no key; prices,
 * context, live availability). If the gateway is unreachable, the registry file plus the
 * pool hashes in Redis, so the pages still render.
 */
export async function loadCatalog(): Promise<Catalog> {
  try {
    const res = await fetch(`${gatewayUrl()}/v1/models`, { cache: "no-store", signal: AbortSignal.timeout(3000) });
    if (res.ok) return { entries: catalogFromGateway(await res.json()), source: "gateway" };
  } catch {
    // fall through to the registry
  }
  const models = await listCatalog();
  const pools = await poolStatuses(models.map((m) => m.id));
  return {
    entries: models.map((model) => ({ model, pool: pools[model.id] ?? { state: "unknown" } })),
    source: "registry",
  };
}
