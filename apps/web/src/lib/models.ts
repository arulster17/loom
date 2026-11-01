import "server-only";

import { readFile } from "node:fs/promises";
import path from "node:path";

import Redis from "ioredis";
import yaml from "js-yaml";

import { catalogFromRegistry, parsePoolHash, type ModelInfo, type PoolStatus } from "@/lib/model-catalog";

/** config/models.yaml: $LOOM_MODELS_YAML, else the repo copy (dev runs from apps/web). */
export function modelsYamlPath(): string {
  return process.env.LOOM_MODELS_YAML ?? path.resolve(process.cwd(), "../../config/models.yaml");
}

let cached: { at: number; models: ModelInfo[] } | undefined;

/** The models the public API offers (priced and not disabled), from the registry. */
export async function listCatalog(): Promise<ModelInfo[]> {
  if (cached && Date.now() - cached.at < 60_000) return cached.models;
  const doc = yaml.load(await readFile(modelsYamlPath(), "utf8"));
  const models = catalogFromRegistry(doc);
  cached = { at: Date.now(), models };
  return models;
}

const g = globalThis as unknown as { __loomRedis?: Redis };

function redis(): Redis {
  if (!g.__loomRedis) {
    g.__loomRedis = new Redis(process.env.LOOM_REDIS_URL ?? "redis://localhost:6379/0", {
      lazyConnect: true,
      maxRetriesPerRequest: 1,
      connectTimeout: 1000,
      enableOfflineQueue: false,
    });
    g.__loomRedis.on("error", () => {}); // reported per call as "unknown" status
  }
  return g.__loomRedis;
}

/**
 * Pool state per model from `loom:pool:<model>` (contracts §10), read-only. If Redis is
 * unreachable every model reads "unknown"; the page still renders.
 */
export async function poolStatuses(modelIds: string[]): Promise<Record<string, PoolStatus>> {
  const out: Record<string, PoolStatus> = Object.fromEntries(modelIds.map((id) => [id, { state: "unknown" }]));
  try {
    const r = redis();
    if (r.status === "wait" || r.status === "end") await r.connect();
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
