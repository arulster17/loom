import { AuthzError, isOrgActive } from "@/lib/authz";
import { getPlaygroundKey } from "@/lib/data/playground";
import { gatewayBody, playgroundRequest } from "@/lib/playground";
import { gatewayUrl } from "@/lib/services";
import { currentOrgContext } from "@/lib/session";

export const dynamic = "force-dynamic";

const MAX_BODY_BYTES = 1_048_576;

function error(status: number, message: string, code: string) {
  return Response.json({ error: { message, type: "invalid_request_error", param: null, code } }, { status });
}

function sameOrigin(req: Request): boolean {
  const origin = req.headers.get("origin");
  if (!origin) return true; // same-origin fetches from older browsers; the cookie is SameSite=Lax anyway
  const host = req.headers.get("x-forwarded-host") ?? req.headers.get("host");
  try {
    return new URL(origin).host === host;
  } catch {
    return false;
  }
}

/**
 * Playground chat: the browser posts here with its session cookie; this handler checks
 * the session and org, then streams from the gateway with the user's server-side
 * playground key (src/lib/data/playground.ts). Nothing of the conversation is stored or
 * logged.
 */
export async function POST(req: Request) {
  if (!sameOrigin(req)) return error(403, "Cross-site request refused.", "forbidden");
  const ctx = await currentOrgContext();
  if (!ctx) return error(401, "Sign in first.", "unauthenticated");
  if (!isOrgActive(ctx)) return error(403, "This account is not active.", "org_not_active");

  const raw = await req.text();
  if (raw.length > MAX_BODY_BYTES) return error(413, "The conversation is too long.", "request_too_large");
  let parsed;
  try {
    parsed = playgroundRequest.safeParse(JSON.parse(raw));
  } catch {
    return error(400, "Invalid JSON.", "invalid_request");
  }
  if (!parsed.success) return error(400, "Invalid playground request.", "invalid_request");

  let key: string;
  try {
    key = await getPlaygroundKey(ctx);
  } catch (e) {
    if (e instanceof AuthzError) return error(403, e.message, e.code);
    throw e;
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${gatewayUrl()}/v1/chat/completions`, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(gatewayBody(parsed.data)),
      signal: req.signal, // the user pressing Stop cancels the upstream request
      cache: "no-store",
    });
  } catch {
    return error(502, "The Loom API is not reachable right now.", "gateway_unreachable");
  }

  const headers = new Headers({ "Cache-Control": "no-store" });
  for (const h of ["content-type", "retry-after", "x-loom-request-id", "x-loom-price-version", "x-loom-max-cost-micros"]) {
    const v = upstream.headers.get(h);
    if (v) headers.set(h, v);
  }
  return new Response(upstream.body, { status: upstream.status, headers });
}
