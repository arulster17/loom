import { beforeEach, describe, expect, it, vi } from "vitest";

import { readBodyLimited } from "@/lib/request-body";

vi.mock("@/lib/session", () => ({ currentOrgContext: vi.fn() }));
vi.mock("@/lib/data/playground", () => ({ getPlaygroundKey: vi.fn() }));

const CAP = 1_048_576;

/** A 16 MiB request body, counting the bytes pulled from it (a reader without a limit
 * pulls all of it). */
function endless(chunk = 64 * 1024, total = 16 * CAP): { stream: ReadableStream<Uint8Array>; pulled: () => number } {
  let pulled = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (pulled >= total) return controller.close();
      pulled += chunk;
      controller.enqueue(new Uint8Array(chunk).fill(0x61));
    },
  });
  return { stream, pulled: () => pulled };
}

function post(body: BodyInit | null, headers: Record<string, string> = {}): Request {
  return new Request("http://localhost:3000/api/playground/chat", {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body,
    duplex: "half",
  } as RequestInit);
}

describe("readBodyLimited", () => {
  it("returns a body within the cap", async () => {
    const r = await readBodyLimited(post('{"a":"é"}'), 100);
    expect(r).toEqual({ ok: true, text: '{"a":"é"}' });
  });

  it("refuses a declared Content-Length above the cap without reading", async () => {
    const { stream, pulled } = endless();
    const r = await readBodyLimited(post(stream, { "content-length": String(CAP + 1) }), CAP);
    expect(r).toEqual({ ok: false });
    expect(pulled()).toBeLessThanOrEqual(64 * 1024); // at most the stream's first buffered chunk
  });

  it("stops reading a chunked body once it passes the cap", async () => {
    const { stream, pulled } = endless();
    const r = await readBodyLimited(post(stream), CAP);
    expect(r).toEqual({ ok: false });
    expect(pulled()).toBeLessThan(CAP + 4 * 64 * 1024);
  });

  it("counts bytes, not characters", async () => {
    expect(await readBodyLimited(post("éé"), 3)).toEqual({ ok: false }); // 4 bytes
    expect(await readBodyLimited(post("éé"), 4)).toEqual({ ok: true, text: "éé" });
  });

  it("returns an empty body as empty text", async () => {
    expect(await readBodyLimited(post(null), 10)).toEqual({ ok: true, text: "" });
  });
});

describe("POST /api/playground/chat body cap", () => {
  beforeEach(async () => {
    const session = await import("@/lib/session");
    vi.mocked(session.currentOrgContext).mockResolvedValue({
      userId: "u",
      orgId: "o",
      orgStatus: "active",
    } as never);
  });

  it("answers 413 to a huge chunked body without buffering it", async () => {
    const { POST } = await import("@/app/api/playground/chat/route");
    const { stream, pulled } = endless();
    const res = await POST(post(stream));
    expect(res.status).toBe(413);
    expect(pulled()).toBeLessThan(CAP + 4 * 64 * 1024);
    expect(await res.json()).toMatchObject({ error: { code: "request_too_large" } });
  }, 10_000);

  it("answers 413 to an oversized Content-Length", async () => {
    const { POST } = await import("@/app/api/playground/chat/route");
    const { stream, pulled } = endless();
    const res = await POST(post(stream, { "content-length": String(CAP + 1) }));
    expect(res.status).toBe(413);
    expect(pulled()).toBeLessThanOrEqual(64 * 1024);
  }, 10_000);
});
