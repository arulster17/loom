// CSRF check for the cookie-authenticated route handlers (POST /api/playground/chat).
// Pure, so it is unit tested; server actions get the equivalent check from Next.js.

/**
 * True when the request's `Origin` is this app's host.
 *
 * - No `Origin`: allowed (same-origin requests from older browsers; the session cookie
 *   is SameSite=Lax, so a cross-site POST carries no session anyway).
 * - `Origin: null` (sandboxed iframes, `data:` pages, some redirects) and malformed
 *   origins, including an empty one: refused.
 * - The host compared is `X-Forwarded-Host` when present (the app behind a proxy or
 *   load balancer), else `Host`. A browser cannot set `X-Forwarded-Host` on a
 *   cross-site request without a CORS preflight, which this app never grants.
 */
export function sameOrigin(headers: Headers): boolean {
  const origin = headers.get("origin");
  if (origin === null) return true; // absent; an empty Origin is malformed, below
  const host = headers.get("x-forwarded-host") ?? headers.get("host");
  try {
    return new URL(origin).host === host;
  } catch {
    return false; // "null" and anything else that is not a URL
  }
}
