import { beforeEach, describe, expect, it, vi } from "vitest";

import { sameOrigin } from "@/lib/same-origin";

vi.mock("@/lib/session", () => ({ currentOrgContext: vi.fn() }));
vi.mock("@/lib/data/playground", () => ({ getPlaygroundKey: vi.fn() }));

const session = await import("@/lib/session");
const playground = await import("@/lib/data/playground");

function h(fields: Record<string, string>): Headers {
  return new Headers(fields);
}

describe("sameOrigin", () => {
  it("accepts the app's own origin, http or https", () => {
    expect(sameOrigin(h({ host: "localhost:3000", origin: "http://localhost:3000" }))).toBe(true);
    expect(sameOrigin(h({ host: "app.loom.dev", origin: "https://app.loom.dev" }))).toBe(true);
  });

  it("accepts a request without Origin (older browsers; the cookie is SameSite=Lax)", () => {
    expect(sameOrigin(h({ host: "localhost:3000" }))).toBe(true);
  });

  it("refuses another site, another port and a look-alike host", () => {
    for (const origin of [
      "https://evil.example",
      "http://localhost:3001",
      "http://localhost",
      "http://localhost:3000.evil.example",
      "http://evil.example/localhost:3000",
    ]) {
      expect(sameOrigin(h({ host: "localhost:3000", origin })), origin).toBe(false);
    }
  });

  it("refuses Origin: null (sandboxed iframes, data: URLs, cross-site redirects)", () => {
    expect(sameOrigin(h({ host: "localhost:3000", origin: "null" }))).toBe(false);
  });

  it("refuses malformed origins, including an empty one", () => {
    for (const origin of ["", " ", "localhost:3000", "not a url", "http://", "://localhost:3000"]) {
      expect(sameOrigin(h({ host: "localhost:3000", origin })), JSON.stringify(origin)).toBe(false);
    }
  });

  it("refuses when the request names no host at all", () => {
    expect(sameOrigin(h({ origin: "http://localhost:3000" }))).toBe(false);
  });

  it("compares against X-Forwarded-Host when a proxy set it, not Host", () => {
    const behindProxy = { host: "web:3000", "x-forwarded-host": "app.loom.dev" };
    expect(sameOrigin(h({ ...behindProxy, origin: "https://app.loom.dev" }))).toBe(true);
    expect(sameOrigin(h({ ...behindProxy, origin: "http://web:3000" }))).toBe(false);
    expect(sameOrigin(h({ ...behindProxy, origin: "https://evil.example" }))).toBe(false);
    // A chain of forwarded hosts is not a single host: refused rather than guessed.
    expect(
      sameOrigin(h({ host: "web:3000", "x-forwarded-host": "app.loom.dev, web:3000", origin: "https://app.loom.dev" })),
    ).toBe(false);
  });
});

describe("POST /api/playground/chat origin check", () => {
  const body = JSON.stringify({
    model: "qwen3-8b",
    messages: [{ role: "user", content: "hi" }],
    temperature: 0.7,
    top_p: 1,
    max_tokens: 16,
  });

  function post(headers: Record<string, string>): Request {
    return new Request("http://localhost:3000/api/playground/chat", {
      method: "POST",
      headers: { "content-type": "application/json", ...headers },
      body,
    });
  }

  beforeEach(() => {
    vi.mocked(session.currentOrgContext).mockReset().mockResolvedValue(null);
    vi.mocked(playground.getPlaygroundKey).mockReset();
  });

  it.each([
    ["a cross-site Origin", { host: "localhost:3000", origin: "https://evil.example" }],
    ["Origin: null", { host: "localhost:3000", origin: "null" }],
    ["a malformed Origin", { host: "localhost:3000", origin: "evil" }],
    ["an Origin that matches Host but not X-Forwarded-Host", { host: "web:3000", "x-forwarded-host": "app.loom.dev", origin: "http://web:3000" }],
  ])("refuses %s with 403 before any session lookup", async (_label, headers) => {
    const { POST } = await import("@/app/api/playground/chat/route");
    const res = await POST(post(headers));
    expect(res.status).toBe(403);
    expect((await res.json()).error.code).toBe("forbidden");
    expect(session.currentOrgContext).not.toHaveBeenCalled();
    expect(playground.getPlaygroundKey).not.toHaveBeenCalled();
  });

  it.each([
    ["the same origin", { host: "localhost:3000", origin: "http://localhost:3000" }],
    ["no Origin", { host: "localhost:3000" }],
    ["the proxy's public origin", { host: "web:3000", "x-forwarded-host": "app.loom.dev", origin: "https://app.loom.dev" }],
  ])("lets %s through to the session check", async (_label, headers) => {
    const { POST } = await import("@/app/api/playground/chat/route");
    const res = await POST(post(headers));
    expect(res.status).toBe(401); // signed out (mocked): got past the origin check
    expect(session.currentOrgContext).toHaveBeenCalledTimes(1);
  });
});
