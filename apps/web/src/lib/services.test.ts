import { afterEach, describe, expect, it, vi } from "vitest";

import { createCheckout, ServiceError } from "./services";

const req = {
  orgId: "6f1c2a4e-1b2c-4d3e-8f90-123456789abc",
  amountMicros: 25_000_000n,
  customerEmail: "a@example.com",
  successUrl: "http://localhost:3000/dashboard/billing?checkout=success",
  cancelUrl: "http://localhost:3000/dashboard/billing?checkout=cancelled",
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("createCheckout (billing's /internal/checkout-sessions)", () => {
  it("posts the contract body with the internal token and returns the pay URL", async () => {
    vi.stubEnv("LOOM_BILLING_URL", "http://billing:8090/");
    vi.stubEnv("LOOM_BILLING_INTERNAL_TOKEN", "tok");
    const fetchMock = vi.fn(async () =>
      Response.json({ id: "cs_test_1", url: "http://localhost:12111/checkout/cs_test_1", amount_micros: 25_000_000 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    expect(await createCheckout(req)).toBe("http://localhost:12111/checkout/cs_test_1");
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("http://billing:8090/internal/checkout-sessions");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer tok");
    expect(JSON.parse(String(init.body))).toEqual({
      org_id: req.orgId,
      amount_micros: 25_000_000,
      customer_email: "a@example.com",
      success_url: req.successUrl,
      cancel_url: req.cancelUrl,
    });
  });

  it("fails as a service error without a token, on errors and on a missing URL", async () => {
    vi.stubEnv("LOOM_BILLING_INTERNAL_TOKEN", "");
    await expect(createCheckout(req)).rejects.toBeInstanceOf(ServiceError);
    vi.stubEnv("LOOM_BILLING_INTERNAL_TOKEN", "tok");
    vi.stubGlobal("fetch", vi.fn(async () => new Response("no", { status: 409 })));
    await expect(createCheckout(req)).rejects.toThrow(/409/);
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ id: "x" })));
    await expect(createCheckout(req)).rejects.toBeInstanceOf(ServiceError);
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("fetch failed"))));
    await expect(createCheckout(req)).rejects.toBeInstanceOf(ServiceError);
  });
});
