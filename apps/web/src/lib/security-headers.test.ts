import { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import nextConfig from "../../next.config";
import { contentSecurityPolicy, newNonce, STATIC_SECURITY_HEADERS } from "@/lib/security-headers";
import { proxy } from "@/proxy";

function directives(csp: string): Record<string, string[]> {
  return Object.fromEntries(
    csp.split(";").map((d) => {
      const [name, ...values] = d.trim().split(/\s+/);
      return [name, values];
    }),
  );
}

const prod = { nonce: "abc123", nextDev: false, loomDev: false };

describe("contentSecurityPolicy", () => {
  it("is strict in production: nonce + strict-dynamic, no unsafe-inline or eval for scripts", () => {
    const d = directives(contentSecurityPolicy(prod));
    expect(d["default-src"]).toEqual(["'self'"]);
    expect(d["script-src"]).toEqual(["'self'", "'nonce-abc123'", "'strict-dynamic'"]);
    expect(d["connect-src"]).toEqual(["'self'"]);
    expect(d["img-src"]).toEqual(["'self'", "data:"]);
    expect(d["object-src"]).toEqual(["'none'"]);
    expect(d["base-uri"]).toEqual(["'self'"]);
    expect(d["frame-ancestors"]).toEqual(["'none'"]);
    expect(d["style-src"]).toEqual(["'self'", "'unsafe-inline'"]);
  });

  it("lets forms (and their redirects) reach only us, Stripe Checkout and Google sign-in", () => {
    expect(directives(contentSecurityPolicy(prod))["form-action"]).toEqual([
      "'self'",
      "https://checkout.stripe.com",
      "https://accounts.google.com",
    ]);
  });

  it("allows the local Stripe stub only with LOOM_ENV=dev", () => {
    const d = directives(contentSecurityPolicy({ ...prod, loomDev: true }));
    expect(d["form-action"]).toEqual(expect.arrayContaining(["http://localhost:*", "http://127.0.0.1:*"]));
    expect(d["script-src"]).not.toContain("'unsafe-eval'");
  });

  it("adds unsafe-eval and the HMR websocket only under next dev", () => {
    const d = directives(contentSecurityPolicy({ ...prod, nextDev: true }));
    expect(d["script-src"]).toContain("'unsafe-eval'");
    expect(d["connect-src"]).toEqual(["'self'", "ws:", "wss:"]);
  });
});

describe("newNonce", () => {
  it("is 128 random bits, base64, different every time", () => {
    const a = newNonce();
    expect(a).toMatch(/^[A-Za-z0-9+/]{22}==$/);
    expect(newNonce()).not.toBe(a);
  });
});

describe("next.config.ts headers()", () => {
  it("sets the constant headers on every path", async () => {
    const rules = await nextConfig.headers!();
    expect(rules).toEqual([{ source: "/:path*", headers: STATIC_SECURITY_HEADERS }]);
    expect(Object.fromEntries(STATIC_SECURITY_HEADERS.map((h) => [h.key, h.value]))).toEqual({
      "X-Frame-Options": "DENY",
      "X-Content-Type-Options": "nosniff",
      "Referrer-Policy": "strict-origin-when-cross-origin",
    });
  });
});

describe("proxy", () => {
  afterEach(() => vi.unstubAllEnvs());

  it("sends a CSP with a fresh nonce, and passes the same CSP to the renderer", () => {
    vi.stubEnv("LOOM_ENV", "prod");
    const a = proxy(new NextRequest("http://localhost:3000/signin"));
    const b = proxy(new NextRequest("http://localhost:3000/signin"));
    const csp = a.headers.get("content-security-policy")!;
    const nonce = /'nonce-([^']+)'/.exec(csp)![1];
    expect(b.headers.get("content-security-policy")).not.toBe(csp);
    // NextResponse.next({ request: { headers } }) forwards overridden request headers so.
    expect(a.headers.get("x-middleware-request-content-security-policy")).toBe(csp);
    expect(a.headers.get("x-middleware-request-x-nonce")).toBe(nonce);
    expect(directives(csp)["form-action"]).not.toContain("http://localhost:*");
    expect(directives(csp)["script-src"]).not.toContain("'unsafe-eval'"); // NODE_ENV=test
  });

  it("reads LOOM_ENV at request time (the image is built once, run as dev or prod)", () => {
    vi.stubEnv("LOOM_ENV", "dev");
    const csp = proxy(new NextRequest("http://localhost:3000/dashboard/billing")).headers.get("content-security-policy")!;
    expect(directives(csp)["form-action"]).toContain("http://localhost:*");
  });
});
