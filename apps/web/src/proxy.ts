import { NextResponse, type NextRequest } from "next/server";

import { contentSecurityPolicy, newNonce } from "@/lib/security-headers";

// Content-Security-Policy with a per-request nonce (src/lib/security-headers.ts). Set on
// the request too: Next.js reads the nonce from the request's CSP header and adds it to
// the scripts it renders. Runs at request time, so LOOM_ENV is the server's, not the
// build's. Authorization is never done here: pages, actions and route handlers check it.

export function proxy(request: NextRequest): NextResponse {
  const nonce = newNonce();
  const csp = contentSecurityPolicy({
    nonce,
    nextDev: process.env.NODE_ENV === "development",
    loomDev: process.env.LOOM_ENV === "dev",
  });
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);
  requestHeaders.set("Content-Security-Policy", csp);
  const response = NextResponse.next({ request: { headers: requestHeaders } });
  response.headers.set("Content-Security-Policy", csp);
  return response;
}

export const config = {
  matcher: [
    {
      // Everything but static files (they carry the constant headers from next.config.ts)
      // and router prefetches (no HTML: nothing to nonce).
      source: "/((?!_next/static|_next/image|favicon.ico).*)",
      missing: [
        { type: "header", key: "next-router-prefetch" },
        { type: "header", key: "purpose", value: "prefetch" },
      ],
    },
  ],
};
