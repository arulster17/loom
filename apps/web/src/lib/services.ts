import "server-only";

// Calls to the other Loom services: the gateway (public API) and billing (checkout).

/** Another service could not be reached or answered badly. */
export class ServiceError extends Error {
  name = "ServiceError";
}

export function gatewayUrl(): string {
  return (process.env.LOOM_GATEWAY_URL ?? "http://localhost:8080").replace(/\/+$/, "");
}

/** The API base URL shown to users in code samples. */
export function publicApiBase(): string {
  return (process.env.LOOM_PUBLIC_API_URL ?? "http://localhost:8080/v1").replace(/\/+$/, "");
}

export function billingUrl(): string {
  return (process.env.LOOM_BILLING_URL ?? "http://localhost:8090").replace(/\/+$/, "");
}

/** Model ids from the gateway's GET /v1/models (OpenAI list shape). */
export async function gatewayModelIds(apiKey: string): Promise<string[]> {
  const res = await fetch(`${gatewayUrl()}/v1/models`, {
    headers: { Authorization: `Bearer ${apiKey}` },
    cache: "no-store",
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`gateway answered ${res.status}`);
  const body = (await res.json()) as { data?: { id?: unknown }[] };
  return (body.data ?? []).map((m) => String(m.id)).filter(Boolean);
}

export interface CheckoutRequest {
  orgId: string;
  amountMicros: bigint;
  customerEmail: string;
  successUrl: string;
  cancelUrl: string;
}

/**
 * Starts a Stripe Checkout for prepaid credit through the billing service:
 * `POST {LOOM_BILLING_URL}/checkout/sessions` (contract proposed by the web app; see
 * apps/web/README.md). Locally the billing service answers with its Stripe stub's URL.
 * Credit is posted only by the signed webhook, never by this call or the success page.
 */
export async function createCheckout(req: CheckoutRequest): Promise<string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (process.env.LOOM_INTERNAL_TOKEN) headers.Authorization = `Bearer ${process.env.LOOM_INTERNAL_TOKEN}`;
  let res: Response;
  try {
    res = await fetch(`${billingUrl()}/checkout/sessions`, {
      method: "POST",
      headers,
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
      body: JSON.stringify({
        org_id: req.orgId,
        amount_micros: Number(req.amountMicros),
        customer_email: req.customerEmail,
        success_url: req.successUrl,
        cancel_url: req.cancelUrl,
      }),
    });
  } catch (e) {
    throw new ServiceError(`billing unreachable: ${e instanceof Error ? e.message : "error"}`);
  }
  if (!res.ok) throw new ServiceError(`billing answered ${res.status}`);
  const body = (await res.json()) as { url?: unknown };
  if (typeof body.url !== "string" || !/^https?:\/\//.test(body.url)) {
    throw new ServiceError("billing returned no checkout URL");
  }
  return body.url;
}
