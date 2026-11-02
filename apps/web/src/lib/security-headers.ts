// Security headers for every response. Pure, so it is unit tested.
//
// - The Content-Security-Policy carries a fresh nonce per request, so it is set by
//   src/proxy.ts at request time. Next.js reads the nonce from the request's CSP header
//   and puts it on its own inline bootstrap scripts and chunk scripts; 'strict-dynamic'
//   then trusts the chunks those load. No 'unsafe-inline' for scripts: an injected
//   <script> has no nonce and does not run. Every page renders per request (the root
//   layout awaits connection()), since a prerendered page would carry no nonce.
// - The other headers are constant and set in next.config.ts for every path, static
//   files included.

export interface CspOptions {
  nonce: string;
  /** `next dev` (NODE_ENV=development): React needs eval for its dev tooling, and HMR a
   * websocket. */
  nextDev: boolean;
  /** LOOM_ENV=dev: the Stripe stub's pay page is on http://localhost:<port>. */
  loomDev: boolean;
}

/**
 * Where a form may submit, including the redirects after a submission (Chrome applies
 * form-action to them): ourselves, then the external pages our server actions redirect
 * to when JavaScript is off (Stripe Checkout; Google sign-in), and the local Stripe stub.
 */
export function formActionSources(loomDev: boolean): string[] {
  const sources = ["'self'", "https://checkout.stripe.com", "https://accounts.google.com"];
  if (loomDev) sources.push("http://localhost:*", "http://127.0.0.1:*");
  return sources;
}

export function contentSecurityPolicy({ nonce, nextDev, loomDev }: CspOptions): string {
  const directives: [string, ...string[]][] = [
    ["default-src", "'self'"],
    ["script-src", "'self'", `'nonce-${nonce}'`, "'strict-dynamic'", ...(nextDev ? ["'unsafe-eval'"] : [])],
    // Inline style attributes and the dev overlay; styles cannot run code.
    ["style-src", "'self'", "'unsafe-inline'"],
    ["img-src", "'self'", "data:"],
    ["font-src", "'self'"],
    ["connect-src", "'self'", ...(nextDev ? ["ws:", "wss:"] : [])],
    ["object-src", "'none'"],
    ["base-uri", "'self'"],
    ["form-action", ...formActionSources(loomDev)],
    ["frame-ancestors", "'none'"],
  ];
  return directives.map((d) => d.join(" ")).join("; ");
}

/** Constant headers for every response (next.config.ts). */
export const STATIC_SECURITY_HEADERS: { key: string; value: string }[] = [
  // frame-ancestors 'none' in the CSP does the same for browsers that support it.
  { key: "X-Frame-Options", value: "DENY" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  // Other sites see only our origin, never a path or query; same-origin requests keep
  // the full URL.
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
];

/** A 128-bit random nonce, base64. */
export function newNonce(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return btoa(String.fromCharCode(...bytes));
}
