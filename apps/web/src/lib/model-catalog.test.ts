import { readFileSync } from "node:fs";
import path from "node:path";

import yaml from "js-yaml";
import { describe, expect, it } from "vitest";

import { catalogFromRegistry, parsePoolHash, poolLabel } from "./model-catalog";

describe("catalog from config/models.yaml", () => {
  const doc = yaml.load(readFileSync(path.resolve(__dirname, "../../../../config/models.yaml"), "utf8"));

  it("lists the servable models with context and prices, as loom_platform.pricing.servable", () => {
    const models = catalogFromRegistry(doc);
    expect(models.map((m) => m.id).sort()).toEqual(["llama-3.3-70b-instruct-fp8", "qwen3-8b"]);
    const qwen = models.find((m) => m.id === "qwen3-8b")!;
    expect(qwen).toMatchObject({ maxContext: 32768, inputPerMtokMicros: 77_000, outputPerMtokMicros: 368_000 });
    expect(qwen.startupNote).toBeNull();
    expect(models.find((m) => m.id.includes("70b"))!.startupNote).toMatch(/20 min/);
  });

  it("skips disabled and unpriced rows", () => {
    const models = catalogFromRegistry({
      models: [
        { id: "a", max_context: 1, pricing: { input_per_mtok: 1, output_per_mtok: 2 }, status: "disabled" },
        { id: "b", max_context: 1, pricing: null, status: "preview" },
        { id: "c", max_context: 8, pricing: { input_per_mtok: 1, output_per_mtok: 2 }, status: "ga" },
      ],
    });
    expect(models.map((m) => m.id)).toEqual(["c"]);
    expect(models[0].cachedInputPerMtokMicros).toBe(1);
  });
});

describe("pool state", () => {
  it("reads the contract hash; ready needs a replica", () => {
    expect(parsePoolHash({ state: "ready", replicas: '[{"id":"p"}]' })).toEqual({ state: "ready" });
    expect(parsePoolHash({ state: "ready", replicas: "[]" }).state).toBe("unavailable");
    expect(parsePoolHash({ state: "warming", retry_after_s: "120" })).toEqual({ state: "warming", retryAfterS: 120 });
    expect(parsePoolHash({ state: "bogus" })).toEqual({ state: "unknown" });
  });

  it("labels states in plain words", () => {
    expect(poolLabel({ state: "ready" }).label).toBe("Ready");
    expect(poolLabel({ state: "warming", retryAfterS: 450 }).detail).toMatch(/about 8 min/);
    expect(poolLabel({ state: "unavailable" }).label).toBe("Temporarily unavailable");
    expect(poolLabel({ state: "cold" }).label).toBe("Asleep");
  });
});
