import { readFileSync } from "node:fs";
import path from "node:path";

import yaml from "js-yaml";
import { describe, expect, it } from "vitest";

import { catalogFromGateway, catalogFromRegistry, parsePoolHash, poolLabel } from "./model-catalog";

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

describe("catalog from the gateway's GET /v1/models", () => {
  // Shape of services/gateway/src/loom_gateway/app.py Gateway.list_models.
  const body = {
    object: "list",
    data: [
      {
        id: "qwen3-8b",
        object: "model",
        created: 0,
        owned_by: "loom",
        display_name: "Qwen3 8B",
        context_length: 32768,
        quantization: "none",
        status: "preview",
        capabilities: { tools: true },
        pricing: {
          currency: "USD",
          price_version: "77000:77000:368000",
          input_per_mtok_micros: 77000,
          cached_input_per_mtok_micros: 77000,
          output_per_mtok_micros: 368000,
          input_per_mtok_usd: "0.077",
          cached_input_per_mtok_usd: "0.077",
          output_per_mtok_usd: "0.368",
        },
        availability: { state: "ready", ready_replicas: 1, retry_after_s: null },
      },
      {
        id: "llama-3.3-70b-instruct-fp8",
        display_name: "Llama 3.3 70B FP8",
        context_length: 32768,
        status: "preview",
        pricing: { input_per_mtok_micros: 571000, cached_input_per_mtok_micros: 571000, output_per_mtok_micros: 797000 },
        availability: { state: "warming", ready_replicas: 0, retry_after_s: 450 },
      },
    ],
  };

  it("reads context, prices and live availability", () => {
    const [qwen, llama] = catalogFromGateway(body);
    expect(qwen.model).toMatchObject({ id: "qwen3-8b", displayName: "Qwen3 8B", maxContext: 32768, outputPerMtokMicros: 368000 });
    expect(qwen.pool).toEqual({ state: "ready", retryAfterS: undefined });
    expect(llama.pool).toEqual({ state: "warming", retryAfterS: 450 });
    expect(llama.model.startupNote).toMatch(/20 min/);
  });

  it("refuses a body without a data list", () => {
    expect(() => catalogFromGateway({ error: {} })).toThrow();
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
